import pyarrow.parquet as pq
import os
import pandas as pd
import numpy as np
import openai
import argparse
from tqdm import tqdm
openai.api_key = ""


def get_completion_messages_idx(df, idx):
    delimiter = "####"
    num_of_questions = min(3, df.iloc[idx]['length'] // 300 + 1)
    wiki_text = df.iloc[idx]['text']
    system_message0 = f"""
    You will be provided with TEXT from wikipedia. \
    The TEXT will be delimited with "####" characters.
    Output a python list of {num_of_questions} json objects, where each object is a multiple choice question whom answers should be in the given TEXT and that has 5 options each.
    The question, question answer options(A, B, C, D, E) should be diverse, broad, challenging, long, detailed and based on the TEXT provided, question should be asked beyond its definition.
    Each question should have one best option(among A, B, C, D, E), while others may be incomplete or wrong. It is desirable that each option would be 2 to 3 sentences long.
    """
    system_message1 = """
    Don't write anything but provide a python list of json objescts in which each json object has the following format 
    {
        'question':"...",
        'A':"...",
        'B':"...",
        'C':"...",
        'D':"...",
        'E':"...",
        "answer":"."
    }
    """
    return [
        {
            'role': 'system',
            'content': system_message0.format(delimiter, num_of_questions) + system_message1
        },
        {
            'role': 'user',
            'content': f"{delimiter}{wiki_text}{delimiter}"
        },
    ]


def get_completion_from_messages(
    messages,
    model="gpt-3.5-turbo-16k",
    temperature=0.5,
    max_tokens=8000
):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        request_timeout=600
    )
    return response.choices[0].message["content"]


def check_valid_response(response):
    try:
        tmp = pd.DataFrame(eval(response))
        assert sum(tmp['question'].isna()) == 0
        assert sum(tmp['A'].isna()) == 0
        assert sum(tmp['B'].isna()) == 0
        assert sum(tmp['C'].isna()) == 0
        assert sum(tmp['D'].isna()) == 0
        assert sum(tmp['E'].isna()) == 0
        assert sum(tmp['answer'].isna()) == 0
        assert sum(tmp['answer'].isin(set(list('ABCDE')))) == len(tmp)
        return True
    except:
        return False


def main(args):
    files = os.listdir('./wikipedia_20220301/')

    df_lists = []
    for file in files:
        tmp = pq.read_table(os.path.join('./wikipedia_20220301/', file))
        df_lists.append(tmp.to_pandas())

    df = pd.concat(df_lists, axis=0).reset_index(drop=True)
    df['length'] = df['text'].apply(lambda x: len(x.split()))

    sub_df = df[df['length'] < 2000].reset_index(drop=True)
    sub_df = sub_df[sub_df['length'] > 100].reset_index(drop=True)

    seen_idx = np.load('./seen_idx.npy')
    sub_df = sub_df.drop(index=seen_idx)
    seen_titles = np.load('./seen_titles_2000.npy', allow_pickle=True)
    sub_df = sub_df[~sub_df['title'].isin(seen_titles)]
    seen_titles = np.load('./seen_titles_5000.npy', allow_pickle=True)
    sub_df = sub_df[~sub_df['title'].isin(seen_titles)]

    pages_count = args.pages_count

    sample_df = sub_df.sample(n=pages_count, random_state=888)

    multiple_choice_questions = []
    seen_titles = []
    max_completion_attempts = 3
    cnt = 0
    for i in tqdm(range(pages_count)):
        messages = get_completion_messages_idx(sample_df, i)
        acc = 0
        while acc < max_completion_attempts:
            try:
                response = get_completion_from_messages(messages)
                assert check_valid_response(response)
                multiple_choice_questions.append(response)
                seen_titles.append(sample_df.iloc[i]['title'])
                cnt += 1
                print(f'Finish one page! Current have finish {cnt} / {i + 1} pages!')
                break
            except:
                acc += 1
        if i % 100 == 0:
            np.save(f"multiple_choice_questions_{pages_count}.npy", multiple_choice_questions)
            np.save(f"seen_titles_{pages_count}.npy", seen_titles)
            
    np.save(f"multiple_choice_questions_{pages_count}.npy", multiple_choice_questions)
    np.save(f"seen_titles_{pages_count}.npy", seen_titles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pages_count', type=int, default=1000)
    args = parser.parse_args()

    main(args)
