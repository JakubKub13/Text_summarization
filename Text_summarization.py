# https://www.kaggle.com/shivamkushwaha/bbc-full-text-document-classification
#!wget -nc https://lazyprogrammer.me/course_files/nlp/bbc_text_cls.csv
#pip install transformers

import pandas as pd
import numpy as np
import textwrap
from transformers import pipeline

data_frame = pd.read_csv('bbc_text_cls.csv')
data_frame.head()

doc = data_frame[data_frame.labels == 'business']['text'].sample(random_state=42)

def wrap(x):
  return textwrap.fill(x, replace_whitespace=False, fix_sentence_endings=True)

print(wrap(doc.iloc[0]))

