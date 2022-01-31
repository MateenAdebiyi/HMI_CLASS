#Question 2

import pandas as pd

df_a =pd.read_csv("metastatic_solid_tumors_a.tsv" , sep="\t")

df_b =pd.read_csv("metastatic_solid_tumors_b.tsv" , sep="\t")

df_c = pd.merge(df_a, df_b)

df_male = df_c[df_c["Sex"] == 'Male']
df_male = df_male[df_male["Age"].between(40,60) ]

df_female = df_c[df_c["Sex"] == 'Female']
df_female = df_female[df_female["Age"].between(40,60) ]

df_male.to_csv("male-patients-40-60.csv")

df_female.to_csv("female-patients-40-60.csv")