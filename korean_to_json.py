import pandas as pd
import numpy as np
import os
import re
import json
import random
from collections import defaultdict



# Intent별 남길 슬롯 정의
intent_slot_restrictions = {
    "음식점_요청": ["메뉴", "수량", "음료", "인원", "재료"],
    "음식점_문의": ["메뉴", "수량", "음료", "인원", "재료"],
    "음식점_질문": ["메뉴", "수량", "인원", "재료"],
    "음식점_요구": ["메뉴", "수량", "음료", "인원", "재료"],
    "음식점_주문": ["메뉴", "수량", "음료", "인원", "재료"],
    "음식점_선택": ["메뉴", "수량", "음료", "인원"],
    "의복의류점_요청": ["사이즈", "색상", "소재", "재질", "제품"],
    "의복의류점_문의": ["사이즈", "색상", "소재", "재질", "제품"],
    "학원_문의": ["과목", "기간", "대상", "반", "성별", "시간", "요일", "횟수"],
    "카페_문의": ["결제수단", "금액", "맛", "메뉴", "수량", "시간", "온도", "장소", "재료", "적립수단", "커피종류", "토핑"],
    "카페_요구": ["메뉴", "사이즈", "수량", "온도", "재료", "토핑"],
    "카페_주문": ["메뉴", "사이즈", "수량", "온도", "토핑"],
    "숙박_문의": ["결제수단", "구이음식종류", "기간", "날짜", "달", "방종류", "사람수", "수량", "숙박시설종류", "시간", "연령대", "요일", "이동수단", "인원", "일자"]
}

# 데이터 로드 및 전처리
def load_and_filter_data(file_path):
    data = pd.read_excel(file_path)
    data = data.drop(data[(data['MAIN'].isnull()) & (data['QA'] == 'Q')].index)
    data = data[data['QA'] == 'Q']
    return data[data["지식베이스"].notna()]

# 지식베이스에서 "/" 뒷부분 없는 데이터를 제거하고, 지식베이스의 단어가 SENTENCE에 없는 경우 데이터에서 제거
def remove_empty_after_slash(data):
    def clean_knowledge_base(knowledge_base):
        if pd.isna(knowledge_base):
            return knowledge_base
        elements = knowledge_base.split(',')
        cleaned_elements = [element for element in elements if not element.endswith('/')]
        return ','.join(cleaned_elements)
    
    def has_matching_terms(row):
        if pd.isna(row['지식베이스']) or pd.isna(row['SENTENCE']):
            return False
        knowledge_terms = [term.split('/')[0] for term in row['지식베이스'].split(',')]
        return any(term in row['SENTENCE'] for term in knowledge_terms)

    data['지식베이스'] = data['지식베이스'].apply(clean_knowledge_base)
    return data[data.apply(has_matching_terms, axis=1)].copy()

# Intent 규칙 생성 함수
def create_category_rules(data):
    unique_main_values = data['MAIN'].dropna().unique()
    return {value[-2:]: [value[-2:]] for value in unique_main_values}

# Intent 분류 및 필터링 함수
def classify_and_filter_intents(data, category_rules):
    def apply_rules(main):
        if pd.isna(main):
            return "기타"
        for category, keywords in category_rules.items():
            if any(keyword in main for keyword in keywords):
                return category
        return "기타"
    
    data['Subcategory'] = data['MAIN'].apply(apply_rules)
    data['Intent'] = data['DOMAIN'] + "_" + data['Subcategory']
    intent_counts = data['Intent'].value_counts()
    intents_above_threshold = intent_counts[intent_counts >= 200].index
    return data[data['Intent'].isin(intents_above_threshold)]


def create_slot(text):
    temp_lists = re.split(",", text)
    temp_lists = [re.split("/", temp_list) for temp_list in temp_lists]
    # 빈 문자열을 제거하고 필요한 경우에만 리스트에 추가
    temp_lists = [temp_list for temp_list in temp_lists if len(temp_list) > 1 and temp_list[-1]]
    for temp_list in temp_lists:
        temp_list.reverse()
    return temp_lists

def aggregate_slot(temp_lists):
    return {temp_list[-1]: temp_list[0] for temp_list in temp_lists if temp_list[-1]}

def extract_top_categories(data_dict_list, min_count):
    slot_values = []
    slot_categories = []
    
    for item in data_dict_list:
        for key, value in item.items():
            slot_values.append(key.strip())
            slot_categories.append(value.strip())

    df = pd.DataFrame({
        'slot_value': slot_values,
        'slot_category': slot_categories
    })
    
    top_categories = df['slot_category'].value_counts()
    filtered_top_categories = top_categories[top_categories >= min_count].index.tolist()
    
    return filtered_top_categories

# 상위 카테고리 목록 기반 슬롯 필터링 함수
def filter_slot_by_top_categories(data, top_categories_list):

    data = data[data['dict'].apply(lambda d: all(value in top_categories_list for value in d.values()) if isinstance(d, dict) else False)]
    return data

# 토큰화된 column 생성
def create_tokenized_column(dataframe_input):
    tokenized = []
    for _, row in dataframe_input.iterrows():
        text = row["SENTENCE"]
        substrings = [key.strip() for key in row["dict"].keys() if key.strip()]
        results = []
        for substring in substrings:
            if substring and substring in text:
                parts = text.split(substring, 1)
                if parts[0].strip():
                    results.extend(parts[0].strip().split())
                results.append(substring)
                text = parts[1]
        if text.strip():
            results.extend(text.strip().split())
        tokenized.append(results)
    dataframe_input["tokenized"] = tokenized
    return dataframe_input

# JSON 데이터 생성
def make_json(row, domain, idx):
    result = []
    sequence_in = []
    slot_mapping_list = []
    txt = row["tokenized"]
    new_dict = {key.strip(): value.strip() for key, value in row["dict"].items()}
    
    for word in txt:
        if word in new_dict:
            split_word = word.split()
            for s_w in split_word:
                if split_word[0] == s_w:
                    sequence_in.append(s_w)
                    slot_mapping_list.append("B_" + new_dict[word])
                else:
                    sequence_in.append(s_w)
                    slot_mapping_list.append("I_" + new_dict[word])
        else:
            sequence_in.append(word)
            slot_mapping_list.append("O")
    
    for name, entity in zip(sequence_in, slot_mapping_list):
        result.append({"text": name, "entity": entity})
    
    return {
        f"{domain}_{idx}": {
            "text": " ".join(sequence_in),
            "domain" : domain,
            "topic" : row['Intent'].split("_")[-1],
            "intent": row['Intent'],
            "slot_out": " ".join(slot_mapping_list),
        }
    }

# Intent별 슬롯 요약 및 JSON 저장
def save_intent_summary(json_data, file_name):
    with open(file_name, 'w', encoding='utf-8') as json_file:
        json.dump({"data": json_data}, json_file, ensure_ascii=False, indent=4)

# 최종 데이터 처리 및 요약
def process_data(file_path, use_restrictions=True, min_count=30):
    data = load_and_filter_data(file_path)
    data = remove_empty_after_slash(data)
    category_rules = create_category_rules(data)
    # 문자열 대체 작업 수행
    data['지식베이스'] = data['지식베이스'].str.replace('방 종류', '방종류') 
    data['지식베이스'] = data['지식베이스'].str.replace('구이음식 종류', '구이음식종류')
    data['지식베이스'] = data['지식베이스'].str.replace('사람 수', '사람수')
    data['지식베이스'] = data['지식베이스'].str.replace('숙박시설 종류', '숙박시설종류')
    data['지식베이스'] = data['지식베이스'].str.replace('커피 종류', '커피종류')
    data = classify_and_filter_intents(data, category_rules)
    data["slot"] = data["지식베이스"].apply(create_slot)
    data["dict"] = data["slot"].apply(aggregate_slot)
    
    intent_counts = data['Intent'].value_counts()
    intent_summary = {}
    json_data = {}

    for intent in intent_counts.index:
        intent_data = data[data['Intent'] == intent].copy()
        
        # intent별 filtered_slot_categories 설정
        if use_restrictions:
            filtered_slot_categories = intent_slot_restrictions.get(intent, [])
        else:
            data_lst = list(intent_data['dict'])
            filtered_slot_categories = extract_top_categories(data_lst, min_count)
        
        # intent별로 필터 적용
        intent_data = filter_slot_by_top_categories(intent_data, filtered_slot_categories)
        intent_data = create_tokenized_column(intent_data)
        
        for idx, row in intent_data.iterrows():
            domain = row['DOMAIN']
            json_data.update(make_json(row, domain, idx))
        
        intent_summary[intent] = len(intent_data)
        

    save_intent_summary(json_data, 'intent_summary_and_data.json')
    return intent_summary, json_data

# 모든 파일을 처리 및 하나의 JSON 파일로 저장
def process_all_files_in_directory(directory_path, use_restrictions=True, min_count=30):
    files = [f for f in os.listdir(directory_path) if f.endswith('.xlsx')]

    for file_name in files:
        file_path = os.path.join(directory_path, file_name)
        intent_summary, json_data = process_data(file_path, use_restrictions, min_count)
        
        # json_file_name을 directory_path와 동일한 경로에 저장
        json_file_name = os.path.join(directory_path, f"{os.path.splitext(file_name)[0]}.json")
        
        # JSON 파일 저장
        save_intent_summary(json_data, json_file_name)
        
        print(f"{json_file_name} 저장 완료.")

# 각 JSON 파일에서 train, valid, test로 분할하여 저장
def split_individual_files(directory_path):
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.json') and not file_name.startswith('korean_'):
            with open(os.path.join(directory_path, file_name), 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                
                # 'data' 키가 없는 경우 건너뜀
                if 'data' not in loaded_data or not isinstance(loaded_data['data'], dict):
                    print(f"Warning: '{file_name}' does not contain a valid dictionary under 'data' key. Skipping...")
                    continue

                data_items = list(loaded_data['data'].values())
                
                # 데이터 분할 (80% train, 10% valid, 10% test)
                random.shuffle(data_items)
                train_split = int(len(data_items) * 0.8)
                valid_split = int(len(data_items) * 0.1) + train_split

                train_data = data_items[:train_split]
                valid_data = data_items[train_split:valid_split]
                test_data = data_items[valid_split:]

                # 분할된 데이터를 각각의 파일로 저장
                base_file_name = os.path.join(directory_path, os.path.splitext(file_name)[0])
                
                # 각각의 train, valid, test 파일을 별도로 저장
                for split_name, split_data in zip(['_train', '_valid', '_test'], [train_data, valid_data, test_data]):
                    split_file_name = f"{base_file_name}{split_name}.json"
                    with open(split_file_name, 'w', encoding='utf-8') as split_file:
                        json.dump({"data": {f"{file_name}_{i}": item for i, item in enumerate(split_data)}}, split_file, ensure_ascii=False, indent=4)
                    print(f"{split_file_name} 파일 저장 완료")

# 각 split 파일들을 합쳐 최종 train, valid, test 파일 생성
def merge_split_files(directory_path):
    data_splits = {'train': [], 'valid': [], 'test': []}

    # 각 split 파일을 불러와 합침
    for file_name in os.listdir(directory_path):
        if file_name.endswith('_train.json'):
            with open(os.path.join(directory_path, file_name), 'r', encoding='utf-8') as f:
                data_splits['train'].extend(json.load(f).get("data", {}).values())
        elif file_name.endswith('_valid.json'):
            with open(os.path.join(directory_path, file_name), 'r', encoding='utf-8') as f:
                data_splits['valid'].extend(json.load(f).get("data", {}).values())
        elif file_name.endswith('_test.json'):
            with open(os.path.join(directory_path, file_name), 'r', encoding='utf-8') as f:
                data_splits['test'].extend(json.load(f).get("data", {}).values())

    # 합쳐진 데이터를 최종 파일로 저장
    for split, split_data in data_splits.items():
        output_file = os.path.join(directory_path, f'korean_{split}.json')
        with open(output_file, 'w', encoding='utf-8') as json_file:
            json.dump({"data": {f"{split}_{i}": item for i, item in enumerate(split_data)}}, json_file, ensure_ascii=False, indent=4)
        print(f"{output_file} 파일 저장 완료")
        
# 디렉터리 경로 설정
directory_path = "./data/raw_data"
# 실행
process_all_files_in_directory(directory_path, use_restrictions=True)  # 개별 JSON 파일 생성
split_individual_files(directory_path)  # 개별 JSON 파일에서 train, valid, test로 분할하여 저장
merge_split_files(directory_path)  # 모든 train, valid, test 파일을 합쳐 최종 파일 생성