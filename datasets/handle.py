import sys
import os
from huggingface_hub import login
from datasets import Dataset
import pandas as pd
import numpy as np

def main():

    # *** CHANGE THESE VALUES ***

    upload_path = 'raw-txt/en-sw.txt'         
    language_abv = 'sw'
    language = "Swahili"
    repo_id = 'yuzhiliu8/Multilingual-orig'

    # ***************************

    with open(upload_path, 'r+', encoding='utf-8') as file:
        header_line = f'English\t{language}\n'
        content = file.readlines()
        original_length = len(content)
        if content[0] != header_line:   #adds language headers to txt file
            file.seek(0)
            file.write(header_line)
            for line in content:
                file.write(line)


    df = pd.read_csv(upload_path, delimiter="\t", encoding='utf-8', on_bad_lines='skip')
    print(df)
    print(f'LINES WITH ERRORS (SKIPPED) {original_length - len(df)}')

    
    push_to_hf(
        dataframe=df,
        repo_id=repo_id,
        language_abbreviation=language_abv
    )
        

def Handle_CSV():
    upload_path = 'raw-txt/SWC-FRA.csv'         

    df = pd.read_csv(upload_path, encoding='utf8')
    df = df.rename(columns={"French":"English"})
    null_row_indexes = df.loc[df['Swahili'].isna()].index
    df = df.drop(null_row_indexes)
    df['English'] = np.nan

    print(df)


def push_to_hf(dataframe, repo_id, language_abbreviation):
    _in = input("Do you wish to upload this CSV to HuggingFace? \nPress Y to continue: ").lower()
    if (_in == "y"):
        train_val_split = int(len(dataframe) * 0.8)
        val_test_split = int(len(dataframe) * 0.9)

        #Create Train, val, test splits
        TRAIN = Dataset.from_pandas(dataframe[:train_val_split])
        VALIDATION = Dataset.from_pandas(dataframe[train_val_split:val_test_split])
        TEST = Dataset.from_pandas(dataframe[val_test_split:])

        #Create train, val, test directory paths

        TRAIN_path = f"train/{language_abbreviation}/"
        VALIDATION_path = f"validation/{language_abbreviation}/"
        TEST_path = f"test/{language_abbreviation}/"
        
        

    
        _in = input(f'''
        INFORMATION:
            REPO_ID: {repo_id}
            LANGUAGE_ABBREVIATION: {language_abbreviation}
            TRAIN dataset length: {len(TRAIN)}
            VAL dataset length: {len(VALIDATION)}
            TEST dataset length: {len(TEST)}
            pushing TRAIN to path: {TRAIN_path}
            pushing VALIDATION to path: {VALIDATION_path}
            pushing TEST to path: {TEST_path}

        if information is correct, press Y to continue: ''').lower()
        if (_in ==  "y"):
            TRAIN.push_to_hub(
                repo_id=repo_id,
                config_name="train",
                split=f'{language_abbreviation}_train',
                data_dir=TRAIN_path
            )
            VALIDATION.push_to_hub(
                repo_id=repo_id,
                config_name="validation",
                split=f'{language_abbreviation}_validation',
                data_dir=VALIDATION_path
            )
            TEST.push_to_hub(
                repo_id=repo_id,
                config_name="test",
                split=f'{language_abbreviation}_test',
                data_dir=TEST_path
            )

        

if __name__ == "__main__":
    main()

