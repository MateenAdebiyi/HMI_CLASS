#Question 1

import pandas as pd
df1 =pd.read_csv("HMI-example-data.tsv" , sep="\t")
#print (df1)

df2 =df1.drop(['Study ID', 'Sample ID' , 'Sample Type'], axis = 1) 

df_male = df2[df2["Sex"] == 'Male']

df_female = df2[df2["Sex"] == 'Female']

df_male = df_male.fillna(df_male.mode().iloc[0])

df_female = df_female.fillna(df_female.mode().iloc[0])

df_male.to_csv("male-patients.csv")

df_female.to_csv("female-patients.csv")