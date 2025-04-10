import os
import json

# 입력 파일과 출력 파일 경로 설정
input_directory = './data/raw_data'
output_directory = './data/amazon+korean'

# 필요한 상위 및 하위 폴더를 생성
def create_output_dirs():  
    os.makedirs(os.path.join(output_directory, 'train'), exist_ok=True)   
    os.makedirs(os.path.join(output_directory, 'dev'), exist_ok=True)   
    os.makedirs(os.path.join(output_directory, 'test'), exist_ok=True)

# JSON 파일에서 text, slot, intent를 추출하여 각각의 파일에 저장
def extract_and_save_data(file_path, split_name):
    # 파일 경로 설정
    output_text_file = os.path.join(output_directory, split_name, 'seq.in')
    output_slot_file = os.path.join(output_directory, split_name, 'seq.out')
    output_intent_file = os.path.join(output_directory, split_name, 'label')
    
    # 데이터 로드
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # "data" 키를 사용하여 실제 데이터에 접근
        if "data" in data:
            data = data["data"]
        else:
            print(f"Warning: 'data' key not found in {file_path}")
            return

    # 파일에 text, slot, intent를 각각 저장
    with open(output_text_file, "w", encoding="utf-8") as text_file, \
         open(output_slot_file, "w", encoding="utf-8") as slot_file, \
         open(output_intent_file, "w", encoding="utf-8") as intent_file:
        for entry in data.values():  # data가 딕셔너리일 경우 values() 사용
            if isinstance(entry, dict) and all(key in entry for key in ["text", "slot_out", "intent"]):
                # 텍스트 저장
                text_file.write(entry["text"] + "\n")
                # 슬롯 저장 (B_, I_ 형식을 B-, I- 형식으로 변경)
                slot_out = entry["slot_out"].replace('B_', 'B-').replace('I_', 'I-')
                slot_file.write(slot_out + "\n")
                # 의도 저장
                intent_file.write(entry["intent"] + "\n")
            else:
                print(f"Warning: Invalid entry format in {file_path}: {entry}")

    print(f"{split_name} 데이터가 성공적으로 저장되었습니다.")

# 모든 JSON 파일을 순회하며 extract_and_save_data 함수 호출
def process_all_json_files():
    # 특정 파일명으로 불러와 모델 input 형식으로 변경, 파일명 변경 가능
    for file_name, split_name in [("amazon+korean_train.json", "train"), ("amazon+korean_valid.json", "dev"), ("amazon+korean_test.json", "test")]:
        file_path = os.path.join(input_directory, file_name)
        if os.path.exists(file_path):
            extract_and_save_data(file_path, split_name)
        else:
            print(f"{file_name} 파일을 찾을 수 없습니다.")

# 모든 intent를 수집하고 중복을 제거하여 intent_label.txt에 저장
def save_unique_intents():
    unique_intents = set()
    for split in ['train', 'dev', 'test']:
        intent_file_path = os.path.join(output_directory, split, 'label')
        with open(intent_file_path, "r", encoding="utf-8") as f:
            for line in f:
                unique_intents.add(line.strip())
    
    # 정렬 후 intent_label.txt에 저장
    with open(os.path.join(output_directory, 'intent_label.txt'), "w", encoding="utf-8") as output_file:
        output_file.write("PAD\n")  # 'PAD' 라인을 맨 앞에 추가
        for intent in sorted(unique_intents):
            output_file.write(intent + "\n")
    
    print("중복되지 않는 의도가 intent_label.txt에 저장되었습니다.")

# 모든 slot을 수집하고 중복을 제거하여 slot_label.txt에 저장
def save_unique_slots():
    unique_slots = set()
    for split in ['train', 'dev', 'test']:
        slot_file_path = os.path.join(output_directory, split, 'seq.out')
        with open(slot_file_path, "r", encoding="utf-8") as f:
            for line in f:
                slots = line.strip().split()
                for slot in slots:
                    if slot != 'O':  # 'O'는 제외
                        unique_slots.add(slot)
    
    # 정렬 후 slot_label.txt에 저장
    with open(os.path.join(output_directory, 'slot_label.txt'), "w", encoding="utf-8") as output_file:
        output_file.write("PAD\nUNK\nO\n")  # 'PAD', 'UNK', 'O' 라인을 맨 앞에 추가
        for slot in sorted(unique_slots):
            output_file.write(slot + "\n")
    
    print("중복되지 않는 슬롯이 slot_label.txt에 저장되었습니다.")

# 전체 실행 함수
def main():
    create_output_dirs()  # 출력 디렉터리 생성 
    process_all_json_files()  # JSON 파일 처리 
    save_unique_intents()  # 고유한 intent 저장  
    save_unique_slots()  # 고유한 slot 저장  
  
# 메인 함수 실행   
if __name__ == "__main__":   
    main()