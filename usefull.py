import pandas as pd
import os
import urllib.request
from pathlib import Path
from transformers import BertTokenizerFast
import torch
import numpy as np


def download_file(url, output_file):
  Path(output_file).parent.mkdir(parents=True, exist_ok=True)
  urllib.request.urlretrieve (url, output_file)

if not os.path.isfile('./datasets/bc5cdr/train.txt') : download_file('https://raw.githubusercontent.com/shreyashub/BioFLAIR/master/data/ner/bc5cdr/train.txt', './datasets/bc5cdr/train.txt')
if not os.path.isfile('./datasets/bc5cdr/test.txt') : download_file('https://raw.githubusercontent.com/shreyashub/BioFLAIR/master/data/ner/bc5cdr/test.txt', './datasets/bc5cdr/test.txt')
if not os.path.isfile('./datasets/bc5cdr/dev.txt') : download_file('https://raw.githubusercontent.com/shreyashub/BioFLAIR/master/data/ner/bc5cdr/dev.txt', './datasets/bc5cdr/dev.txt')


def read_conll(filename):
    df = pd.read_csv(filename,
                    sep = '\t', header = None, keep_default_na = False,
                    names = ['words', 'pos', 'chunk', 'labels'],
                    quoting = 3, skip_blank_lines = False)
    df = df[~df['words'].astype(str).str.startswith('-DOCSTART- ')] # Remove the -DOCSTART- header
    df['sentence_id'] = (df.words == '').cumsum()
    print(df[df.words != ''])
    return df[df.words != '']


def align_label(texts, labels):
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids

class DataSequence(torch.utils.data.Dataset):

    def __init__(self, df):
        lb = [i.split() for i in df['labels'].values.tolist()]
        txt = df['text'].values.tolist()
        self.texts = [tokenizer(str(i),
                               padding='max_length', max_length = 512, truncation=True, return_tensors="pt") for i in txt]
        self.labels = [align_label(i,j) for i,j in zip(txt, lb)]
    def __len__(self):
        return len(self.labels)

    def get_batch_data(self, idx):
        return self.texts[idx]

    def get_batch_labels(self, idx):
        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):
        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)
        return batch_data, batch_labels
      
def preprocessConll2003(datasetPath = "./datasets/conll2003/conll2003.csv"):
  df = pd.read_csv(datasetPath)
  # Split labels based on whitespace and turn them into a list
  labels = [i.split() for i in df['labels'].values.tolist()]
  unique_labels = set()
  for lb in labels:
    [unique_labels.add(i) for i in lb if i not in unique_labels]
  # print(len(unique_labels))
  
  # Map each label into its id representation and reciprocally
  labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
  ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}
  reformatedDf = []
  for index, row in df.iterrows():
    wordList = row["text"].split()
    labelList = row["labels"].split()
    if len(wordList) == len(labelList):
      for i, w in enumerate(wordList):
        reformatedDf.append((w, labelList[i]))
  df = pd.DataFrame(reformatedDf, columns=["words", "labels"]).drop_duplicates()
  df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(.8 * len(df)), int(.9 * len(df))])
  # print(df_train, df_val, df_test )
  return df_train, df_val, df_test 




  

# preprocessConll2003()

def reassembleDataset(datasetPath, sep = "\t"):
    with open(datasetPath, "r") as file :
        sentence = []
        labels = []
        newDataset = {}
        for line in file:
            line = line.strip()
            if line != "":
                word, label = line.split(sep )
                sentence.append(word)
                labels.append(label)
                
            else :
                newDataset[" ".join(sentence)] = " ".join(labels)
                sentence = []
                labels = []
                
        if len(sentence) > 0 : newDataset[" ".join(sentence)] = " ".join(labels)
        df = pd.DataFrame.from_dict({"text": newDataset.keys(), "labels": newDataset.values()})
        df.to_csv(datasetPath[: datasetPath.rfind(".")] + "_assembled.csv", sep=sep, index=False)
            
        
        