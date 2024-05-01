# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1deDprMVQojgSsmTUEkyWVhnCixG7Ks1E
"""

import pandas as pd
data_path = "/content/drive/MyDrive/reranked_data.csv"
data = pd.read_csv(data_path)

print(data['content'].iloc[1])

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("suriya7/bart-finetuned-text-summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("suriya7/bart-finetuned-text-summarization")

def generate_summary(text):
    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_new_tokens=100, do_sample=False)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary



summary2 = generate_summary(data['content'].iloc[1])
print(summary2)

data['summary'] = data['content'].apply(generate_summary)



csv_path = '/content/drive/My Drive/summarized_ranked.csv'
data.to_csv(csv_path, index=False)