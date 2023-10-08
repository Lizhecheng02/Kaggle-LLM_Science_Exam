# import pandas as pd
# import random

# df1 = pd.read_json("./Clean Dataset/30k full no #.json")
# # df2 = pd.read_json("./Clean Dataset/60k full no #.json")
# # df3 = pd.read_json("./Clean Dataset/50k full no #.json")
# df4 = pd.read_json("./Clean Dataset/36k full no #.json")
# df5 = pd.read_json("./Clean Dataset/25k 0914 no #.json")
# df = pd.concat([df1, df4, df5])

# df = df.reset_index(drop=True)
# df_shuffle = df.copy()


# def shuffle_answer(idx):
#     try:
#         tmp = ['A', 'B', 'C', 'D', 'E']
#         random.shuffle(tmp)
#         hm = {k: v for k, v in zip(['A', 'B', 'C', 'D', 'E'], tmp)}
#         hm_b = {v: k for k, v in zip(['A', 'B', 'C', 'D', 'E'], tmp)}
#         df_shuffle.loc[idx, ['A', 'B', 'C', 'D', 'E']] = [
#             df.loc[idx, hm[k]] for k in ['A', 'B', 'C', 'D', 'E']]
#         df_shuffle.loc[idx, 'answer'] = hm_b[df.loc[idx, 'answer']]
#     except:
#         print("Error!")


# for i in range(len(df_shuffle)):
#     shuffle_answer(i)

# df = df_shuffle.copy()

# print(df["answer"].value_counts())
# print(df.shape)

# df.reset_index(drop=True, inplace=True)
# df.sample(len(df))

# df.to_json("./Clean Dataset/93k full no #.json", orient="records")

import pandas as pd
import random

df = pd.read_csv("Raw Dataset/MMLU_17k_with_context2.csv")

df.drop(columns=["is_question"], axis=1, inplace=True)
df.reset_index(drop=True, inplace=True)
df.insert(6, "E", "")
print(df.head())

df = df.reset_index(drop=True)
df_shuffle = df.copy()


def shuffle_answer(idx):
    try:
        tmp = ['A', 'B', 'C', 'D', 'E']
        random.shuffle(tmp)
        hm = {k: v for k, v in zip(['A', 'B', 'C', 'D', 'E'], tmp)}
        hm_b = {v: k for k, v in zip(['A', 'B', 'C', 'D', 'E'], tmp)}
        df_shuffle.loc[idx, ['A', 'B', 'C', 'D', 'E']] = [
            df.loc[idx, hm[k]] for k in ['A', 'B', 'C', 'D', 'E']]
        df_shuffle.loc[idx, 'answer'] = hm_b[df.loc[idx, 'answer']]
    except:
        print("Error!")


for i in range(len(df_shuffle)):
    shuffle_answer(i)

df = df_shuffle.copy()

print(df["answer"].value_counts())
print(df.shape)

df.reset_index(drop=True, inplace=True)
df.sample(len(df))

df.to_json("./Raw Dataset/17k mmlu.json", orient="records")
