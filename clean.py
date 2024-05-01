import pandas as pd


def clean_df(fn):
    df = pd.read_csv(fn)
    print (df.head(10))
    filtered_df = df[df["content"].str.len() >= 4000]
    filtered_df['rank'] = filtered_df.groupby(['gl', 'searchTerms']).cumcount() + 1
    print (filtered_df.head(10))
    filtered_df = filtered_df[['gl', 'searchTerms', 'rank', 'title', 'snippet', 'link', 'content']]
    filtered_df.to_csv("reranked_data.csv")
    return filtered_df

fn = "updated_file.csv"
df = clean_df(fn)
