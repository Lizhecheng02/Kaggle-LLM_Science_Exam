import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv('Combined Dataset.csv')
train_df = train_df.dropna()
train_df = train_df[:8347]

for train_column in train_df.columns:
    if train_column != 'id':
        train_df[train_column] = train_df[train_column].astype('string')


def cal_length(text):
    return len(text.split(' '))


train_df['total_length'] = train_df['prompt'].apply(cal_length) + train_df['A'].apply(cal_length) + \
    train_df['B'].apply(cal_length) + train_df['C'].apply(cal_length) + \
    train_df['D'].apply(cal_length) + train_df['E'].apply(cal_length)

train_df = train_df[train_df['total_length'] <= 256]
# train_df = train_df.sample(len(train_df), random_state=2023)
train_df.drop(columns=['total_length'], inplace=True)

df = train_df.copy()

train_df = pd.read_csv('claude2.csv')
print(train_df.head())
train_df = train_df.dropna()

for train_column in train_df.columns:
    if train_column != 'id':
        train_df[train_column] = train_df[train_column].astype('string')


def cal_length(text):
    return len(text.split(' '))


train_df['total_length'] = train_df['prompt'].apply(cal_length) + train_df['A'].apply(cal_length) + \
    train_df['B'].apply(cal_length) + train_df['C'].apply(cal_length) + \
    train_df['D'].apply(cal_length) + train_df['E'].apply(cal_length)

print(train_df.sort_values(by='total_length'))

train_df = train_df[train_df['total_length'] <= 256]

print(len(train_df))

train_df['total_length'].plot(kind='hist')
plt.show()

print(train_df['total_length'].value_counts().sort_index(ascending=True))

unique_train_df = train_df.drop_duplicates(
    subset=['prompt', 'A', 'B', 'C', 'D', 'E'])

unique_train_df.drop(columns=['total_length'], inplace=True)

total_df = pd.concat([df, unique_train_df])
total_df['id'] = range(len(total_df))

# print(len(unique_train_df))
total_df.to_csv('High Quality Dataset.csv', index=False)
