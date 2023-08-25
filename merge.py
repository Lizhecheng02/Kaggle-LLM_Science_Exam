import pandas as pd

df1 = pd.read_csv('15k Dataset.csv')
df2 = pd.read_csv('6000_all_categories_questions.csv')

df2.insert(0, "id", range(len(df2)))
df2.to_csv('6000_all_categories_questions.csv', index=False)

df = pd.concat([df1, df2])
df['id'] = range(len(df))

df.to_csv('21k Dataset.csv', index=False)
