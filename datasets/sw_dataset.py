import pandas as pd

df = pd.read_csv("txt_files/en-sw.txt", sep="\s{4,} | \t").head(500)
print(df)



