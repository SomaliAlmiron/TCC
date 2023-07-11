import pandas as pd
from sklearn import preprocessing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("inputPath", help="arquivo input Tcc", type=str)
parser.add_argument("outputPath", help="arquivo output Tcc", type=str)
args = parser.parse_args()

df = pd.read_csv(args.inputPath)

df.drop(['contry_of_res','used_app_before', 'age_desc', 'relation', "Class/ASD", "ethnicity","result" ], axis = 1, inplace = True)

df = df.rename(columns={"age": 'idade',
                        "gender":"genero",
                        "austim": "autismo",
                        "jundice": "ictericia",
                        "A1_Score": "A1",
                        "A2_Score": "A2",
                        "A3_Score": "A3",
                        "A4_Score": "A4",
                        "A5_Score": "A5",
                        "A6_Score": "A6",
                        "A7_Score": "A7",
                        "A8_Score": "A8",
                        "A9_Score": "A9",
                        "A10_Score": "A10",
})

dicionario = {
   'no':0,
   'yes':1}
dicionario2 = {
   'NO':0,
   'YES':1}

df = df.replace({'ictericia': dicionario})
df = df.replace({'autismo': dicionario})

df = df.drop(df[df.idade == '?'].index)
df.idade = df.idade.astype(int)
df = pd.get_dummies(df)
df.to_csv(args.outputPath, index=False)