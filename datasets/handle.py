from datasets import Dataset
import pandas as pd
from huggingface_hub import HfApi
import os

def main():

    # *** CHANGE THESE VALUES ***

    upload_path = 'raw-txt/en-my.txt'         
    language_abv = 'my'
    repo_id = 'yuzhiliu8/Multilingual-orig'

    api = HfApi()

    if "train" not in os.listdir():
        os.mkdir("train")
        os.mkdir("validation")
        os.mkdir("test")

    # ***************************

    # with open(upload_path, 'r+', encoding='utf-8') as file:
    #     header_line = f'English\tText\n'
    #     content = file.readlines()
    #     original_length = len(content)
    #     if content[0] != header_line:   #adds language headers to txt file
    #         file.seek(0)
    #         file.write(header_line)
    #         for line in content:
    #             file.write(line)

    chunksize = 20000000

    chunk_number = 0
    df = pd.read_csv(upload_path, delimiter="\t", encoding='utf-8', on_bad_lines='skip', engine="python", header=None, names=['English', 'Text'])
    print(df)
    

    push_to_hf(
        dataframe=df,
        repo_id=repo_id,
        language_abbreviation=language_abv,
        ch_number=chunk_number,
        api = api,
    )
        

  


def push_to_hf(dataframe, repo_id, language_abbreviation, ch_number, api):
    train_val_split = int(len(dataframe) * 0.8)
    val_test_split = int(len(dataframe) * 0.9)

    #Create Train, val, test splits
    TRAIN = dataframe[:train_val_split]
    VALIDATION = dataframe[train_val_split:val_test_split]
    TEST = dataframe[val_test_split:]

    TRAIN_FILENAME = f'{language_abbreviation}_train-' + (5 - len(str(ch_number))) * "0" + str(ch_number) + ".parquet"
    VALIDATION_FILENAME = f'{language_abbreviation}_validation-' + (5 - len(str(ch_number))) * "0" + str(ch_number) + ".parquet"
    TEST_FILENAME = f'{language_abbreviation}_test-' + (5 - len(str(ch_number))) * "0" + str(ch_number) + ".parquet"

    TRAIN_local_path = f'train/{TRAIN_FILENAME}'
    VALIDATION_local_path = f'validation/{VALIDATION_FILENAME}'
    TEST_local_path = f'test/{TEST_FILENAME}'

    TRAIN.to_parquet(TRAIN_local_path)
    VALIDATION.to_parquet(VALIDATION_local_path)
    TEST.to_parquet(TEST_local_path)

    TRAIN_repo_path = f"train/{language_abbreviation}/{TRAIN_FILENAME}"
    VALIDATION_repo_path = f"validation/{language_abbreviation}/{VALIDATION_FILENAME}"
    TEST_repo_path = f"test/{language_abbreviation}/{TEST_FILENAME}"
    

    print(f'''
    INFORMATION:
        REPO_ID: {repo_id}
        LANGUAGE_ABBREVIATION: {language_abbreviation}
        TRAIN dataset length: {len(TRAIN)}
        VAL dataset length: {len(VALIDATION)}
        TEST dataset length: {len(TEST)}
        pushing TRAIN to path: {TRAIN_repo_path}
        pushing VALIDATION to path: {VALIDATION_repo_path}
        pushing TEST to path: {TEST_repo_path}
''')
    api.upload_file(
        path_or_fileobj=TRAIN_local_path,
        path_in_repo=TRAIN_repo_path,
        repo_id=repo_id,
        repo_type='dataset'
        )
    api.upload_file(
        path_or_fileobj=VALIDATION_local_path,
        path_in_repo=VALIDATION_repo_path,
        repo_id=repo_id,
        repo_type='dataset'
        )
    api.upload_file(
        path_or_fileobj=TEST_local_path,
        path_in_repo=TEST_repo_path,
        repo_id=repo_id,
        repo_type='dataset'
        )
    os.remove(TRAIN_local_path)
    os.remove(VALIDATION_local_path)
    os.remove(TEST_local_path)
            # TRAIN.push_to_hub(
            #     repo_id=repo_id,
            #     config_name="train",
            #     split=f'{language_abbreviation}_train',
            #     data_dir=TRAIN_path
            # )
            # VALIDATION.push_to_hub(
            #     repo_id=repo_id,
            #     config_name="validation",
            #     split=f'{language_abbreviation}_validation',
            #     data_dir=VALIDATION_path
            # )
            # TEST.push_to_hub(
            #     repo_id=repo_id,
            #     config_name="test",
            #     split=f'{language_abbreviation}_test',
            #     data_dir=TEST_path
            # )
            

        

if __name__ == "__main__":
    main()


