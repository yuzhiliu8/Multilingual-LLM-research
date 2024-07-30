from datasets import load_dataset, Dataset
import pandas as pd

def modify():

    # ********* UPDATE THESE *********
    parquet_path = 'Multilingual-orig/validation/sw/sw_validation-00000-of-00001.parquet'
    save_path = 'Multilingual-orig/validation/sw/sw_validation-UPDATED.parquet'



    df = pd.read_parquet(parquet_path)
    df.columns = ["English", "Text"]
    print(df)
    cont = input(f'Saving this df as a parquet in PATH: {save_path}\n Press Y to continue: ').lower()
    if (cont == 'y'):
        df.to_parquet(save_path)



if __name__ == "__main__":
    modify()