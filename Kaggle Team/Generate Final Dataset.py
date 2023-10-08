import pandas as pd

# df_205k = pd.read_json("./Clean Dataset/205k full no #.json")
# df_205k = df_205k.fillna("")
# print("The shape of df_205k:", df_205k.shape)
# df_mmlu = pd.read_json("./Clean Dataset/17k mmlu no #.json")
# df_mmlu = df_mmlu.fillna("")
# print("The shape of df_mmlu:", df_mmlu.shape)

# df_222k = pd.concat([df_205k, df_mmlu])
# df_222k.drop_duplicates(subset=["prompt", "A", "B", "C", "D", "E"])
# print("The shape of df_222k:", df_222k.shape)
# print(df_222k["answer"].value_counts())

# df_222k.reset_index(drop=True, inplace=True)

# df_222k.to_json("./Clean Dataset/222k full no #.json", orient="records")

# df_222k = pd.read_json("./Clean Dataset/222k full no #.json")
# df_222k.sample(len(df_222k), random_state=42)
# df_222k.reset_index(drop=True, inplace=True)

# single_chunk_len = int(len(df_222k) / 5)

# df_chunk1 = df_222k[: 3 * single_chunk_len]
# df_chunk1.reset_index(drop=True, inplace=True)
# print("The shape of chunk1:", df_chunk1.shape)

# df_chunk2 = df_222k[single_chunk_len: 4 * single_chunk_len]
# df_chunk2.reset_index(drop=True, inplace=True)
# print("The shape of chunk2:", df_chunk2.shape)

# df_chunk3 = df_222k[single_chunk_len * 2:]
# df_chunk3.reset_index(drop=True, inplace=True)
# print("The shape of chunk3:", df_chunk3.shape)

# df_chunk4 = pd.concat(
#     [df_222k[3 * single_chunk_len:], df_222k[:single_chunk_len]]
# )
# df_chunk4.reset_index(drop=True, inplace=True)
# print("The shape of chunk4:", df_chunk4.shape)

# df_chunk5 = pd.concat(
#     [df_222k[4 * single_chunk_len:], df_222k[: 2 * single_chunk_len]]
# )
# df_chunk5.reset_index(drop=True, inplace=True)
# print("The shape of chunk5:", df_chunk5.shape)

# df_chunk1.to_json("./Clean Dataset/133k fold1 (1,2,3).json", orient="records")
# df_chunk2.to_json("./Clean Dataset/133k fold2 (2,3,4).json", orient="records")
# df_chunk3.to_json("./Clean Dataset/133k fold3 (3,4,5).json", orient="records")
# df_chunk4.to_json("./Clean Dataset/133k fold4 (4,5,1).json", orient="records")
# df_chunk5.to_json("./Clean Dataset/133k fold5 (5,1,2).json", orient="records")

# df_222k = pd.read_json("./Clean Dataset/222k full no #.json")
# df_222k.sample(len(df_222k), random_state=42)
# df_222k.reset_index(drop=True, inplace=True)

# single_chunk_len = int(len(df_222k) / 4)

# df_chunk1 = df_222k[: 2 * single_chunk_len]
# df_chunk1.reset_index(drop=True, inplace=True)
# print("The shape of chunk1:", df_chunk1.shape)

# df_chunk2 = df_222k[single_chunk_len: 3 * single_chunk_len]
# df_chunk2.reset_index(drop=True, inplace=True)
# print("The shape of chunk2:", df_chunk2.shape)

# df_chunk3 = df_222k[2 * single_chunk_len:]
# df_chunk3.reset_index(drop=True, inplace=True)
# print("The shape of chunk3:", df_chunk3.shape)

# df_chunk4 = pd.concat(
#     [df_222k[3 * single_chunk_len:], df_222k[:single_chunk_len]]
# )
# df_chunk4.reset_index(drop=True, inplace=True)
# print("The shape of chunk4:", df_chunk4.shape)

# df_chunk1.to_json("./Clean Dataset/111k fold1 (1,2).json", orient="records")
# df_chunk2.to_json("./Clean Dataset/111k fold2 (2,3).json", orient="records")
# df_chunk3.to_json("./Clean Dataset/111k fold3 (3,4).json", orient="records")
# df_chunk4.to_json("./Clean Dataset/111k fold4 (4,1).json", orient="records")

df_222k = pd.read_json("./Clean Dataset/222k full no #.json")
df_race = pd.read_json("./Clean Dataset/Race no #.json")
df_truth = pd.read_json("./Clean Dataset/TruthQA no #.json")
df_222k = pd.concat([df_222k, df_race, df_truth])

df_222k.sample(len(df_222k), random_state=2023)
df_222k.reset_index(drop=True, inplace=True)
df_222k.to_json("./Final Dataset/303k.json", orient="records")

single_chunk_len = int(len(df_222k) / 3)

df_chunk1 = df_222k[: 2 * single_chunk_len]
df_chunk1.reset_index(drop=True, inplace=True)
print("The shape of chunk1:", df_chunk1.shape)
print("The answer distribution of chunk1:", df_chunk1["answer"].value_counts())

df_chunk2 = df_222k[single_chunk_len:]
df_chunk2.reset_index(drop=True, inplace=True)
print("The shape of chunk2:", df_chunk2.shape)
print("The answer distribution of chunk2:", df_chunk2["answer"].value_counts())

df_chunk3 = pd.concat(
    [df_222k[2 * single_chunk_len:], df_222k[:single_chunk_len]]
)
df_chunk3.reset_index(drop=True, inplace=True)
print("The shape of chunk3:", df_chunk3.shape)
print("The answer distribution of chunk3:", df_chunk3["answer"].value_counts())

df_chunk1.to_json("./Final Dataset/202k fold1 (1,2).json", orient="records")
df_chunk2.to_json("./Final Dataset/202k fold2 (2,3).json", orient="records")
df_chunk3.to_json("./Final Dataset/202k fold3 (3,4).json", orient="records")
