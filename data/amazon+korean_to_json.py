import pandas as pd
from ast import literal_eval
import re
from datasets import load_dataset
import inspect
from sklearn.model_selection import train_test_split

intents_dict={
    'datetime_query':'날짜_및_시간_조회',
    'iot_hue_lightchange':'IoT_조명_변경',
    'transport_ticket':'교통_티켓',
    'takeaway_query':'테이크아웃_문의',
    'qa_stock':'주식_문의',
    'general_greet':'일반_인사',
    'recommendation_events':'이벤트_추천',
    'music_dislikeness':'음악_비호감',
    'iot_wemo_off':'IoT_Wemo_끄기',
    'cooking_recipe':'요리_레시피',
    'qa_currency':'환율_문의',
    'transport_traffic':'교통_혼잡도',
    'general_quirky':'일반_독특함',
    'weather_query':'날씨_문의',
    'audio_volume_up':'오디오_볼륨_올리기',
    'email_addcontact':'이메일_연락처_추가',
    'takeaway_order':'테이크아웃_주문',
    'email_querycontact':'이메일_연락처_조회',
    'iot_hue_lightup':'IoT_조명_밝게_하기',
    'recommendation_locations':'장소_추천',
    'play_audiobook':'오디오북_재생',
    'lists_createoradd':'목록_생성_또는_추가',
    'news_query':'뉴스_문의',
    'alarm_query':'알람_조회',
    'iot_wemo_on':'IoT_Wemo_켜기',
    'general_joke':'일반_농담',
    'qa_definition':'정의_문의',
    'social_query':'소셜_문의',
    'music_settings':'음악_설정',
    'audio_volume_other':'오디오_볼륨_기타_설정',
    'calendar_remove':'캘린더_삭제',
    'iot_hue_lightdim':'IoT_조명_어둡게_하기',
    'calendar_query':'캘린더_조회',
    'email_sendemail':'이메일_보내기',
    'iot_cleaning':'IoT_청소',
    'audio_volume_down':'오디오_볼륨_낮추기',
    'play_radio':'라디오_재생',
    'cooking_query':'요리_문의',
    'datetime_convert':'날짜_및_시간_변환',
    'qa_maths':'수학_문의',
    'iot_hue_lightoff':'IoT_조명_끄기',
    'iot_hue_lighton':'IoT_조명_켜기',
    'transport_query':'교통_조회',
    'music_likeness':'음악_호감',
    'email_query':'이메일_문의',
    'play_music':'음악_재생',
    'audio_volume_mute':'오디오_볼륨_음소거',
    'social_post':'소셜_게시',
    'alarm_set':'알람_설정',
    'qa_factoid':'사실_문의',
    'calendar_set':'캘린더_설정',
    'play_game':'게임_재생',
    'alarm_remove':'알람_삭제',
    'lists_remove':'목록_제거',
    'transport_taxi':'택시',
    'recommendation_movies':'영화_추천',
    'iot_coffee':'IoT_커피',
    'music_query':'음악_문의',
    'play_podcasts':'팟캐스트_재생',
    'lists_query':'목록_조회'
}

slot_dict={
    'sport_type':'운동_유형',
    'food_type':'음식_유형',
    'place_name':'장소_이름',
    'device_type':'장치_유형',
    'music_album':'음악_앨범',
    'currency_name':'화폐_이름',
    'definition_word':'정의_단어',
    'time':'시간',
    'transport_type':'교통_유형',
    'person':'사람',
    'business_name':'비즈니스_이름',
    'general_frequency':'일반_빈도',
    'player_setting':'플레이어_설정',
    'radio_name':'라디오_이름',
    'personal_info':'개인_정보',
    'ingredient':'재료',
    'event_name':'이벤트_이름',
    'playlist_name':'재생목록_이름',
    'song_name':'노래_이름',
    'movie_type':'영화_유형',
    'movie_name':'영화_이름',
    'coffee_type':'커피_유형',
    'drink_type':'음료_유형',
    'transport_descriptor':'교통_설명자',
    'audiobook_name':'오디오북_이름',
    'house_place':'집_장소',
    'transport_agency':'교통_기관',
    'date':'날짜',
    'music_genre':'음악_장르',
    'business_type':'비즈니스_유형',
    'game_type':'게임_유형',
    'game_name':'게임_이름',
    'podcast_descriptor':'팟캐스트_설명',
    'cooking_type':'요리_유형',
    'email_folder':'이메일_폴더',
    'meal_type':'식사_유형',
    'podcast_name':'팟캐스트_이름',
    'email_address':'이메일_주소',
    'app_name':'앱_이름',
    'order_type':'주문_유형',
    'transport_name':'교통_이름',
    'color_type':'색상_유형',
    'weather_descriptor':'날씨_설명',
    'change_amount':'변경_금액',
    'time_zone':'시간대',
    'joke_type':'농담_유형',
    'news_topic':'뉴스_주제',
    'media_type':'미디어_유형',
    'timeofday':'시간대',
    'alarm_type':'알람_유형',
    'list_name':'목록_이름',
    'music_descriptor':'음악_설명',
    'artist_name':'아티스트_이름',
    'audiobook_author':'오디오북_저자',
    'relation':'관계'
}

pattern=r'\[(.*?)\]'


def parse_text(text):## > '[date :금요일]','[time :오전 아홉 시]','에','깨워줘'
    pattern=re.compile(r'\[.*?\]|\S+')
    matches=pattern.findall(text)
    return matches

# raw amazon dataset intent 가 정수:int 형태로 되어있음 -> str로 변환
def matching_intent(idx_intent, intents_dict):
    intent_keys = list(intents_dict.keys())  
    intent_key = intent_keys[idx_intent]  
    intent_str = intents_dict[intent_key]
    return intent_str


def slot_extract(annot_utt,slot_dict):
    result=[]
    slot_entity_name=[]
    slot_entity_element=[]
    
    parsed_list=parse_text(annot_utt)
    
    for word in parsed_list:
        if ":" in word:
            word=word[1:-1]
            
            entity_name=word.split(':')[0].strip()
            entity_name_ko=slot_dict[entity_name]
            entity_element=word.split(':')[1].strip().split() # ['오후','아홉시']
            for i in entity_element:   
                if i == entity_element[0]:
                    slot_entity_name.append("B_" + entity_name_ko)
                    slot_entity_element.append(i)
                else:
                    slot_entity_name.append("I_" + entity_name_ko)
                    slot_entity_element.append(i)
        else:
            entity_name="O"
            entity_element=word
            slot_entity_name.append(entity_name)
            slot_entity_element.append(entity_element)

    for name,element in zip(slot_entity_name,slot_entity_element):
        result.append({"text":element,"entity":name})
        
    output_slot=" ".join(slot_entity_name)
    return result,output_slot

def making_data_structure(idx):
    intent_class=matching_intent(idx['intent'],intents_dict)
    
    slot_result,output=slot_extract(idx['annot_utt'],slot_dict)
    
    return {
        idx['id']:{
            "text" :idx['utt'],
            "intent" :intent_class,
            "slot_out" :output,
            "slot_mapping" :slot_result 
        }
    }
    
# ["slot_method"]["slot"] None filtering and delete unchanged_translation
def filter_empty_slots(dataset):
    filtered_dataset=dataset.filter(
        lambda x:x['slot_method']['slot'] != [] and 'unchanged_translation' not in x['slot_method']['method']
    )
    return filtered_dataset

# data -> dataframe
def make_dataframe(data):
    results = []
    for idx in data:
        row = making_data_structure(idx)
        for k, v in row.items():
            v['id'] = k
            results.append(v)

    df = pd.DataFrame(results)
    return df

# create intent slot dictionary 
def intent_slot_map(df):
    intent_slot_dict={}
    for i,row in df.iterrows():
        word_slot=[w['entity'][2:] for w in row['slot_mapping'] if w['entity'] != 'O']
        
        if row['intent'] not in intent_slot_dict:
            intent_slot_dict[row['intent']]=[]
            
        for entity in word_slot:
            if entity not in intent_slot_dict[row['intent']]:
                intent_slot_dict[row['intent']].append(entity)


    return intent_slot_dict 

# compare slots between two slot dictionaries.
def compare_slots(train_slot, test_slot):
    all_intents = set(train_slot.keys()).union(set(test_slot.keys()))
    differences = {}

    for intent in all_intents:
        train_entities = set(train_slot.get(intent, []))
        test_entities = set(test_slot.get(intent, []))

        test_only = test_entities - train_entities
        if test_only:
            differences[intent] = list(test_only)

    return differences


def to_json_dict(df:pd.DataFrame):
    """
    dataframe -> "a" : {"b": {"c": ???, 'd': ???}} format
    """
    import inspect
    caller_locals = inspect.currentframe().f_back.f_locals

    for arg_name in caller_locals:
        if caller_locals[arg_name] is df and any(keyword in arg_name for keyword in ['train', 'test', 'valid','val']):
            suffix = arg_name.split('_')[-1]
            if suffix == 'val':
                suffix = 'valid'

    print(suffix)
    json_dict = {}
    for i, row in df.iterrows():
        json_dict[f"{suffix}_{i}"] = {}
        json_dict[f"{suffix}_{i}"]["text"] = row["text"]
        json_dict[f"{suffix}_{i}"]["intent"] = row["intent"]
        json_dict[f"{suffix}_{i}"]["slot_out"] = row["slot_out"]
        

    return {"data":json_dict}

def split_dataset(df:pd.DataFrame):
    from sklearn.model_selection import train_test_split
    """
    Split df into Train / Test / Val in (8:1:1)
    """
    df_total = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df_train, df_rest = train_test_split(df_total, test_size=0.2, random_state=42)
    df_val, df_test = train_test_split(df_rest, test_size=0.5, random_state=42)
    
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    return df_train, df_test, df_val

# extract indices which's only in 'df_any' but not in 'df_train'
# Test / Valid에는 있는데 Train에는 없는 의도 데이터포인트 제거를 위해, 이러한 index return
def test_only_idx(df_train, df_any):

    caller_locals = inspect.currentframe().f_back.f_locals

    for arg_name in caller_locals:
        if caller_locals[arg_name] is df_any and arg_name.startswith("df_"):
            suffix = arg_name.split('_', 1)[-1]
            
            # print(f"Suffix of df_any: {test_suffix}")

    # 기존 코드
    train_slot = intent_slot_map(df_train)
    test_slot = intent_slot_map(df_any)
    
    test_only_slot = compare_slots(train_slot, test_slot)
    test_only_slot_idx = []

    for intent, slots in test_only_slot.items():
        filtered_df = df_any[df_any['intent'] == intent]
        for i, row in filtered_df.iterrows():
            slot_entities = [x['entity'][2:] for x in row['slot_mapping'] if x['entity'] != 'O']
            for slot in slots:
                if slot in slot_entities:
                    # print(intent, ':', slot)
                    test_only_slot_idx.append(i)

    test_only_slot_idx = sorted(list(set(test_only_slot_idx)))
    # print(f"{suffix}_only_slot_idx:", test_only_slot_idx)
    return test_only_slot_idx

# 한글 데이터셋 'text', 'intent' , 'slot_out', 'slot_mapping' 칼럼 만들기
def matching_cols(df):

    df2 = df.copy()
    kor_slot_mapping = []
    for i,row in df.iterrows():
        kor_slot_mapping_list = []
        texts = row['text'].split(' ')
        slots = row['slot_out'].split(' ')
        for t,s in zip(texts,slots):
            kor_slot_mapping_dict = {}
            kor_slot_mapping_dict['text'] = t
            kor_slot_mapping_dict['entity'] = s
            # print(kor_slot_mapping_dict)
            kor_slot_mapping_list.append(kor_slot_mapping_dict)
    
        
        kor_slot_mapping.append(kor_slot_mapping_list)
    df2['slot_mapping'] = kor_slot_mapping
    df2 = df2.drop(columns=['domain','topic'])
    return df2

# Train, Test, Valid 합쳐서 중복 row 제거 후 모드별로 Return
def drop_duplicated_by_mode(df_train,df_test,df_val):
    """
    argument train,test,val 순서 맞춰주세요
    """
    df_train['mode'] = 'train'
    df_test['mode'] = 'test'
    df_val['mode'] = 'val'
    df = pd.concat([df_train,df_test,df_val],axis=0)
    df_dropped = df[~df['text'].duplicated()]
    df_train_dropped = df_dropped[df_dropped['mode'] == 'train']
    df_test_dropped = df_dropped[df_dropped['mode'] == 'test']
    df_val_dropped = df_dropped[df_dropped['mode'] == 'val']
    before_df_list = [df_train,df_test,df_val]
    after_df_list = [df_train_dropped, df_test_dropped, df_val_dropped]
    print('중복제거')
    for before_df, after_df in zip(before_df_list,after_df_list):

        print(f" {before_df['mode'].unique()[0]} : {len(before_df)} -> {len(after_df)}")

    return df_train_dropped, df_test_dropped, df_val_dropped
    
# 이후부터는 필요한 경우에 사용
#########################################################################
# 시간, 시간대 태그 들어간 data들 수작업 진행

# df = pd.concat([df_train,df_val,df_test],axis=0)


def time_timezone_extract(df,save_path):

    """
    return df_time(시간 태그가 들어간), df_timzone(시간대 태그가 들어간)
    """
    time_idx = []
    for i, row in df.iterrows():
        slots = [x['entity'][2:] for x in row['slot_mapping'] if x['entity'] != 'O']
        for slot in slots:
            if slot == '시간':
                time_idx.append(i)
    time_idx = sorted(list(set(time_idx)))
    
    timezone_idx = []
    for i, row in df.iterrows():
        slots = [x['entity'][2:] for x in row['slot_mapping'] if x['entity'] != 'O']
        for slot in slots:
            if slot == '시간대':
                timezone_idx.append(i)
    
    timezone_idx = sorted(list(set(timezone_idx)))

    df_time = df.loc[time_idx]
    df_time.to_csv(os.path.join(save_path,'Amazon_kr_시간_dataframe.csv'),encoding='UTF-8')
    
    df_timezone = df.loc[timezone_idx]
    df_timezone.to_csv(os.path.join(save_path,'Amazon_kr_시간대_dataframe.csv'),encoding='UTF-8')
    
    print(f"{os.path.join(save_path,'Amazon_kr_시간_dataframe.csv')} saved.")
    print(f"{os.path.join(save_path,'Amazon_kr_시간대_dataframe.csv')} saved.")

    return df_time, df_timezone

# 수작업한 데이터 불러와서 'slot_mapping'의 entity와 'slot_out'칼럼 matching
def update_slot_mapping(row):
    slots = row['slot_out'].split(' ')
    
    for i, slot in enumerate(slots):
        row['slot_mapping'][i]['entity'] = slot
    
    return row
#########################################################################




#########################################################################
#### 정보(intent slot dictionary 등등...) 추출 모듈 ####
def intent_unique(df):
    return df['intent'].nunique()

def slot_unique(df):
    slot_list = []
    for row in df['slot_out']:
        slots = row.split(' ')
        for slot in slots:
            slot_list.append(slot.split('_',1)[-1])

    slot_set = sorted(set(slot_list))

    return slot_set
        
        
# create word - entity dictionary
def word_entity_dict(df):
    word_entity={}

    for idx,row in df.iterrows():
    
        text_list=[w['entity'] for w in row['slot_mapping']]
        word_list=[w['text'] for w in row['slot_mapping']]
    
        for word,text in zip(word_list,text_list):
            if text != 'O':
                if word in word_entity:
                    if text not in word_entity[word]:
                        word_entity[word].append(text)
                else:
                    word_entity[word]=[text]

    return word_entity


# make new intent to slot dict
def create_intent_slot_dict(df):
    intent_slot_dict = {}
    for i,row in df.iterrows():
        if row['intent'] not in intent_slot_dict.keys():
            intent_slot_dict[row['intent']] = []
        for slot in [x['entity'] for x in row['slot_mapping']]:
            if slot != 'O' and '_'.join(slot.split('_')[1:]) not in intent_slot_dict[row['intent']]:
                intent_slot_dict[row['intent']].append('_'.join(slot.split('_')[1:]))
        

    return intent_slot_dict
# with open('word_entity_dict.json','w') as f:
    # json.dump(word_entity,f,ensure_ascii=False,indent=4)


# update intent-slot dict
def update_intent_slot_dict(df,intent_slot_dict):
    # intent_slot_dict = {}
    for i,row in df.iterrows():
        if row['intent'] not in intent_slot_dict.keys():
            intent_slot_dict[row['intent']] = []
        for slot in [x['entity'] for x in row['slot_mapping']]:
            if slot != 'O' and '_'.join(slot.split('_')[1:]) not in intent_slot_dict[row['intent']]:
                intent_slot_dict[row['intent']].append('_'.join(slot.split('_')[1:]))
        

    return intent_slot_dict
# intent_slot_dict = create_intent_slot_dict(df_train)
# intent_slot_dict = update_intent_slot_dict(df_test,intent_slot_dict)
# intent_slot_dict = update_intent_slot_dict(df_val,intent_slot_dict)

az_massive_train=load_dataset("AmazonScience/massive","ko-KR",split='train')
az_massive_valid=load_dataset("AmazonScience/massive","ko-KR",split='validation')
az_massive_test=load_dataset("AmazonScience/massive","ko-KR",split='test')

filtered_train=filter_empty_slots(az_massive_train)
filtered_valid=filter_empty_slots(az_massive_valid)
filtered_test=filter_empty_slots(az_massive_test)

df_train = make_dataframe(filtered_train)
df_val = make_dataframe(filtered_valid)
df_test = make_dataframe(filtered_test)

test_only_slot_idx = test_only_idx(df_train,df_test)
val_only_slot_idx = test_only_idx(df_train,df_val)

df_test = df_test.drop(labels=test_only_slot_idx)
df_val = df_val.drop(labels=val_only_slot_idx)

# df_train.to_json('./data/raw_data/Amazon_massive_kor_slot_preprocessed_train.json',orient='records',force_ascii=False,indent=4)
# df_test.to_json('./data/raw_data/Amazon_massive_kor_slot_preprocessed_test.json',orient='records',force_ascii=False,indent=4)
# df_val.to_json('./data/raw_data/Amazon_massive_kor_slot_preprocessed_valid.json',orient='records',force_ascii=False,indent=4)

amazon_df_train = df_train.drop(columns='id')
amazon_df_test = df_test.drop(columns='id')
amazon_df_val = df_val.drop(columns='id')
amazon_df_total = pd.concat([amazon_df_train,amazon_df_test,amazon_df_val],axis=0).reset_index(drop=True)
from korean_to_json import *

with open('./data/raw_data/korean_train.json','r') as f:
    data = json.load(f).get('data','').values()
    kor_df_train = pd.DataFrame(data)

with open('./data/raw_data/korean_test.json','r') as f:
    data = json.load(f).get('data','').values()
    kor_df_test = pd.DataFrame(data)

with open('./data/raw_data/korean_valid.json','r') as f:
    data = json.load(f).get('data','').values()
    kor_df_val = pd.DataFrame(data)


kor_df_train = matching_cols(kor_df_train)
kor_df_test = matching_cols(kor_df_test)
kor_df_val = matching_cols(kor_df_val)

kor_df_total = pd.concat([kor_df_train,kor_df_test,kor_df_val],axis=0)
    
merged_df_train = pd.concat([amazon_df_train,kor_df_train],axis=0).reset_index(drop=True)
merged_df_test = pd.concat([amazon_df_test,kor_df_test],axis=0).reset_index(drop=True)
merged_df_val = pd.concat([amazon_df_val,kor_df_val],axis=0).reset_index(drop=True)

df_total = pd.concat([merged_df_train, merged_df_test, merged_df_val],axis=0).reset_index(drop=True)

df_train, df_test, df_val = split_dataset(df_total)

dataset = {'Amazon_massive':{'intent' : intent_unique(amazon_df_total),
                             'slot' :  len(slot_unique(amazon_df_total)),
                             '최종데이터' : len(amazon_df_total)},
          'AI Hub 한국어 대화':{'intent':intent_unique(kor_df_total),
                                'slot':len(slot_unique(kor_df_total)),
                                  '최종데이터': len(kor_df_total)},
          '최종 Train':{'intent' : intent_unique(df_train) ,
                             'slot' : len(slot_unique(df_train)),
                             '최종데이터' : len(df_train)},
            '최종 Valid':{'intent' : intent_unique(df_val),
                             'slot' : len(slot_unique(df_val)),
                             '최종데이터' : len(df_val)},
        '최종 Test':{'intent' : intent_unique(df_test),
                             'slot' : len(slot_unique(df_test)),
                             '최종데이터' : len(df_test)}}

print(pd.DataFrame(dataset).T)
 
dataset_info= pd.DataFrame(dataset).T
dataset_info = dataset_info.reset_index().rename(columns={'index':'Dataset'})                     
                             
train_json = to_json_dict(df_train)
test_json = to_json_dict(df_test)
val_json = to_json_dict(df_val)

directory_path = './data/raw_data'
dataset_info.to_csv(os.path.join(directory_path,'dataset_info.csv'))

with open(os.path.join(directory_path,'amazon+korean_train.json'),'w',encoding='UTF-8') as f:
    json.dump(train_json,f,ensure_ascii=False, indent=4)

with open(os.path.join(directory_path,'amazon+korean_test.json'),'w',encoding='UTF-8') as f:
    json.dump(test_json,f,ensure_ascii=False, indent=4)

with open(os.path.join(directory_path,'amazon+korean_valid.json'),'w',encoding='UTF-8') as f:
    json.dump(val_json,f,ensure_ascii=False, indent=4)

print(f'Saving Total_train(Test,val).json in "{directory_path}" Done')

# df_train.to_json(os.path.join(directory_path,'Total_train.json'),orient='records',force_ascii=False,indent=4)
# df_test.to_json(os.path.join(directory_path,'Total_test.json'),orient='records',force_ascii=False,indent=4)
# df_val.to_json(os.path.join(directory_path,'Total_valid.json'),orient='records',force_ascii=False,indent=4)
 