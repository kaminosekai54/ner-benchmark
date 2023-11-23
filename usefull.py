import pandas as pd
import os
import urllib.request
from pathlib import Path
from transformers import BertTokenizerFast
import torch
import numpy as np
import chardet
import seaborn as sns
import matplotlib.pyplot as plt

def download_file(url, output_file):
  Path(output_file).parent.mkdir(parents=True, exist_ok=True)
  urllib.request.urlretrieve (url, output_file)

if not os.path.isfile('./datasets/bc5cdr/train.txt') : download_file('https://raw.githubusercontent.com/shreyashub/BioFLAIR/master/data/ner/bc5cdr/train.txt', './datasets/bc5cdr/train.txt')
if not os.path.isfile('./datasets/bc5cdr/test.txt') : download_file('https://raw.githubusercontent.com/shreyashub/BioFLAIR/master/data/ner/bc5cdr/test.txt', './datasets/bc5cdr/test.txt')
if not os.path.isfile('./datasets/bc5cdr/dev.txt') : download_file('https://raw.githubusercontent.com/shreyashub/BioFLAIR/master/data/ner/bc5cdr/dev.txt', './datasets/bc5cdr/dev.txt')


def read_bc5CDR(filename):
    df = pd.read_csv(filename,
                    sep = '\t', header = None, keep_default_na = False,
                    names = ['words', 'pos', 'chunk', 'labels'],
                    quoting = 3, skip_blank_lines = False)
    df = df[~df['words'].astype(str).str.startswith('-DOCSTART- ')] # Remove the -DOCSTART- header
    df['sentence_id'] = (df.words == '').cumsum()
    print(df[df.words != ''])
    return df[df.words != '']


def convertBc5CDRToCONLL(filename):
    df = pd.read_csv(filename,
                    sep = '\t', header = None, keep_default_na = False,
                    names = ['words', 'pos', 'chunk', 'labels'],
                    quoting = 3, skip_blank_lines = False)
    df = df[~df['words'].astype(str).str.startswith('-DOCSTART- ')] # Remove the -DOCSTART- header
    finalDf = df[["words", "labels"]]
    finalDf.to_csv(filename.replace(".tsv", ".tmp"), sep="\t", index=False, header=False)
    with open(filename.replace(".tsv", ".tmp"), 'r') as input_file, open(filename, 'w') as output_file:
        for i, line in enumerate(input_file):
            if line.strip("\n").strip(" ") == '\t' and i == 0: continue
            if line.strip("\n").strip(" ") == '\t': output_file.write('\n')
            else: output_file.write(line)
            
    os.remove(filename.replace(".tsv", ".tmp"))


def readDatasetWithSentenceId(datasetName, type):
    df = pd.read_csv(f'datasets/{datasetName}/{type}.tsv', sep = '\t', header = None, keep_default_na = False, names = ['words', 'labels'], quoting = 3, skip_blank_lines = False)
    df = df[~df['words'].astype(str).str.startswith('-DOCSTART- ')] # Remove the -DOCSTART- header
    df['sentence_id'] = (df.words == '').cumsum()
    df['sentence_id'] = df['sentence_id'] +1
    # print(df[df.words != ''])
    return df[df.words != '']
    
# readDatasetWithSentenceId("bc5cdr", "train")
# read_bc5CDR("datasets/bc5cdr/original/train.txt")
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
    result = ""
    with open(datasetPath, "rb") as file :
        result = chardet.detect(file.read())
    with open(datasetPath, "r", encoding=result['encoding']) as file :
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
        df.to_csv(datasetPath[: datasetPath.rfind(".")] + "_assembled.tsv", sep=sep, index=False)
            
        
        
        
def assembleAllDataSet():
    for dataset in os.listdir("datasets/"):
        print(dataset)
        subpath= "datasets/" + dataset + "/"
        fileList = os.listdir(subpath)
        if not f'{dataset}_assembled' in fileList:
            os.makedirs(subpath + dataset+"_assembled/")
            for file in fileList:
                if file.endswith(".tsv") :
                    print(file)
                    reassembleDataset(subpath + file)
                    os.renames(subpath + file.replace (".tsv", "_assembled.tsv"), subpath + dataset+"_assembled/" + file.replace (".tsv", "_assembled.tsv"))
                
                
                

def plot_metrics(df, model_name, dataset_name, metric, show = True):
    results_folder = "results/"
    dataset_folder = f'{results_folder}{dataset_name}/'
    figures_folder = f'{dataset_folder}figures/'
    
    # Create directories if they don't exist
    for folder in [results_folder, dataset_folder, figures_folder]:
        if not os.path.isdir(folder): os.makedirs(folder)
    
    # Extract column names starting  with '_accuracy'
    metric_columns = [col for col in df.columns if col.startswith(f'{metric}_')]

    # Set up Seaborn with a diverging color palette
    sns.set(style="whitegrid")
    colors = sns.color_palette("RdYlGn", len(metric_columns))

    # Plotting the bar chart with seaborn
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=metric_columns, y=df.iloc[0][metric_columns], hue=metric_columns, palette=colors)

    # Customize plot appearance
    plt.title(f'{model_name} {metric} Comparison for {dataset_name}', fontsize=16)
    plt.xlabel('Entity type', fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.ylim(0, 110)  # Set y-axis limit to percentages (0-100)

    # Rotate x-axis labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

    # Add legend
    plt.legend([col.replace(f'{metric}_', "") for col in metric_columns], title="Labels List", loc="upper left")

    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.7)

    # Increase tick label font size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Save the plot with tight layout to prevent label overlaps
    plt.tight_layout()
    if show == True : plt.show()
    plt.savefig(f'{figures_folder}{model_name}_{metric}_plot_for_{dataset_name}.png')
