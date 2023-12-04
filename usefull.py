import pandas as pd
import os
import urllib.request
from pathlib import Path
import numpy as np
import chardet
import seaborn as sns
import matplotlib.pyplot as plt
import json
from sklearn.metrics import classification_report, accuracy_score

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
                

def plot_metrics(df, model_name, dataset_name, metric, sufixe, color, show = True):
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
    if color == None:
        ax = sns.barplot(x=metric_columns, y=df.iloc[0][metric_columns], hue=metric_columns, palette=colors)
    else:
        ax = sns.barplot(x=metric_columns, y=df.iloc[0][metric_columns], color=color)

    # Customize plot appearance
    plt.title(f'{model_name} {sufixe} {metric} Comparison for {dataset_name}', fontsize=16)
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
    plt.savefig(f'{figures_folder}{model_name}_{sufixe}_{metric}_plot_for_{dataset_name}.png')

def generateExempleFile(datasetName, fileType, nbExemplePerClass=2):
    exemples = []
    dfFull = pd.read_csv(f'datasets/{datasetName}/{datasetName}_assembled/{fileType}_assembled.tsv', sep="\t")
    dfFull.labels = dfFull.labels.apply(lambda x : [str(word) for word in str(x).split()])
    dfWords = readDatasetWithSentenceId(datasetName, type=fileType)
    for label in dfWords.labels.unique():
        subDf = dfFull[dfFull.labels.apply(lambda x: label in x)].sample(n=nbExemplePerClass)
        for sentence in subDf.text.values.tolist():
            exemples.append({"text":sentence})
        
    with open(f'datasets/{datasetName}/examples.json', "w") as exempleFile : json.dump(exemples, exempleFile, indent=4)
# Step 2: Identify unique labels
    unique_labels = dfWords['labels'].unique()

    # Step 3: Find rows where the same word has two different labels
    result_rows = []

    for label in unique_labels:
        sub_df = dfWords[dfWords['labels'] == label]
    
        # Group by words to find pairs
        word_groups = sub_df.groupby('words')['labels'].apply(set)
    
        for word, labels in word_groups.items():
            if len(labels) >= 2:
                result_rows.extend(sub_df[sub_df['words'] == word].itertuples(index=False))

    # Convert the result to a new DataFrame
    result_df = pd.DataFrame(result_rows)

    # Display the result
    # print(result_df)
    
# for datasetName in os.listdir("datasets/"):
    # print(datasetName)
    # generateExempleFile(datasetName, "train", nbExemplePerClass=1)

def correctLabels(datasetName, modelName):
    df1 = readDatasetWithSentenceId(datasetName, "test")
    df2 = pd.read_csv(f'results/{datasetName}/{datasetName}_{modelName }_predictions.csv')
    df1.labels= df1.labels.str.replace("I-", "").str.replace("B-", "")
    df2.labels= df2.labels.str.replace("I-", "").str.replace("B-", "")
    custom_labels = list(set(list(df1.labels.unique()) + list(df2.labels.unique())))
    report = classification_report(df1.labels, df2.labels, labels=custom_labels, target_names=custom_labels, output_dict=True)
    data = {
        "modelName": [modelName ],
        "dataset": [datasetName],
        "accuracy_global": [round(report['weighted avg']['precision']*100, 2)],
        "recall_global": [round(report['weighted avg']['recall']*100, 2)],
        "f1_score_global": [round(report['weighted avg']['f1-score'] *100, 2)]
    }
    for label in custom_labels:
        data[f"accuracy_{label}"] = [round(report[label]['precision']*100, 2)]
        data[f"recall_{label}"] = [round(report[label]['recall']*100, 2)]
        data[f"f1_score_{label}"] = [round(report[label]['f1-score']*100, 2)]

    df = pd.DataFrame(data)
    df.to_csv(f'results/{datasetName}/{datasetName}_{modelName }_eval_corrected.csv', sep=",", index=False)
    return df

def evalPerf(datasetName, modelName):
    df1 = readDatasetWithSentenceId(datasetName, "test")
    df2 = pd.read_csv(f'results/{datasetName}/{datasetName}_{modelName }_predictions.csv')
    custom_labels = list(set(list(df1.labels.unique()) + list(df2.labels.unique())))
    report = classification_report(df1.labels, df2.labels, labels=custom_labels, target_names=custom_labels, output_dict=True)
    data = {
        "modelName": [modelName ],
        "dataset": [datasetName],
        "accuracy_global": [round(report['weighted avg']['precision']*100, 2)],
        "recall_global": [round(report['weighted avg']['recall']*100, 2)],
        "f1_score_global": [round(report['weighted avg']['f1-score'] *100, 2)]
    }
    for label in custom_labels:
        data[f"accuracy_{label}"] = [round(report[label]['precision']*100, 2)]
        data[f"recall_{label}"] = [round(report[label]['recall']*100, 2)]
        data[f"f1_score_{label}"] = [round(report[label]['f1-score']*100, 2)]

    df = pd.DataFrame(data)
    df.to_csv(f'results/{datasetName}/{datasetName}_{modelName}_eval.csv', sep=",", index=False)
    return df


def getEncoding(filePath):
    with open(filePath, "rb") as file :
        return chardet.detect(file.read())['encoding']
    
def getGlobalModelResults(modelName):
    dfList = []
    for dataset in [folder for folder in os.listdir("results/") if os.path.isdir(f'results/{folder}')]:
        for file in os.listdir(f'results/{dataset}/'):
            if modelName in file and file.endswith("eval.csv"):
                tmpDf = pd.read_csv(f'results/{dataset}/{file}')
                tmpDf = tmpDf[["modelName", "dataset", "accuracy_global", "recall_global", "f1_score_global"]]
                dfList.append(tmpDf)

    finalDf = pd.concat(dfList, ignore_index=True)
    return finalDf


def getGlobalCorrectedModelResults(modelName):
    dfList = []
    for dataset in [folder for folder in os.listdir("results/") if os.path.isdir(f'results/{folder}')]:
        for file in os.listdir(f'results/{dataset}/'):
            if modelName in file and file.endswith("eval_corrected.csv"):
                tmpDf = pd.read_csv(f'results/{dataset}/{file}')
                tmpDf = tmpDf[["modelName", "dataset", "accuracy_global", "recall_global", "f1_score_global"]]
                dfList.append(tmpDf)

    finalDf = pd.concat(dfList, ignore_index=True)
    return finalDf



def plotGlobalMetricsInOnePlot(df, modelList, metricList, suffixe ="", show=True, color_mapping={'bc5cdr': '#1f78b4', 'jnlpba': '#6a3d9a', 'ncbi-disease': '#ff7f00', 'ontonote': '#e31a1c', 'species800': '#fdbf6f'}):

    # Set up figure and axis
    fig, ax = plt.subplots(nrows=len(modelList), ncols=len(metricList), figsize=(15, 10))

    # Plot the specified metric for each dataset across models
    for i, model in enumerate(modelList):
        modelData = df[df.modelName == model]
        for j, metric in enumerate(metricList):
            metricData = modelData[["dataset", metric]]
            x = np.arange(len(metricData))

            for k, dt in enumerate(metricData.dataset):
                color = color_mapping.get(dt, 'gray')
                datasetData = metricData[metricData.dataset == dt]
                ax[i, j].bar(x[k], datasetData[metric], label=dt, color=color, alpha=0.7)

    # Set labels, title, and legend
    for i, ax_row in enumerate(ax):
        for j, ax_col in enumerate(ax_row):
            ax_col.set_xlabel('Datasets')
            ax_col.set_ylabel(str.upper(metricList[j]).replace("_", " "))
            ax_col.set_xticks(np.arange(len(df.dataset.unique())))
            ax_col.set_xticklabels(df.dataset.unique().tolist(), rotation=45)
            ax_col.set_title(f'{modelList[i]} - {str.upper(metricList[j]).replace("_", " ")}')

    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5, title='Datasets')

    # Show the plot
    plt.suptitle(f'Comparison of global {suffixe} evaluation metrics on all datasets for each model', y=1.05, fontsize=16)
    plt.tight_layout()
    plt.savefig(f'results/global_{suffixe}_Metric_Evaluation.jpg')
    if show == True : plt.show()



def describeDataSet(datasetName):
    description = f'{str.upper(datasetName)} : \n'
    trainDf = readDatasetWithSentenceId(datasetName, "train")
    devDf = rseadDatasetWithSentenceId(datasetName, "dev")
    testDf = readDatasetWithSentenceId(datasetName, "test")
    fullDf = pd.concat([trainDf, devDf, testDf], ignore_index=True)
    train_assembled = pd.read_csv(f'datasets/{datasetName}/{datasetName}_assembled/train_assembled.tsv', sep="\t")
    dev_assembled = pd.read_csv(f'datasets/{datasetName}/{datasetName}_assembled/dev_assembled.tsv', sep="\t")
    test_assembled = pd.read_csv(f'datasets/{datasetName}/{datasetName}_assembled/test_assembled.tsv', sep="\t")
    full_assembled = pd.concat([train_assembled, dev_assembled, test_assembled], ignore_index=True)
    description += f'This dataset is composed of 3 file, \n'
    description += f'Training : \n{len(trainDf)} words for a total of {len(train_assembled)} sentences. \n'
    description += f'Dev : \n{len(devDf)} words for a total of {len(dev_assembled)} sentences. \n'  
    description +=  f'Testing : \n{len(testDf)} words for a total of {len(test_assembled)} sentences. \n'  
    description +=  f'Which represent in total {len(fullDf)} words for a total of {len(full_assembled)} sentences. \n'
    description +=  f'It covers the following entity labels : \n'
    for label in fullDf.labels.unique():
        description += f'{label} : {len(fullDf[fullDf.labels == label])} entities.\n'
    description += "the highest sota f1-score for this data set is at "
    return description

# models_to_plot = ['bert-based', 'Bio-bert-based', 'spark-nlp']
# metric_to_plot = ['f1_score_global', "recall_global", "accuracy_global"]
# df1 = getGlobalModelResults("bert-based")
# df2 = getGlobalModelResults("Bio-bert-based")
# df3 = getGlobalModelResults("spark-nlp")
# df = pd.concat([df1, df2, df3], ignore_index=True).drop_duplicates()
# df.to_csv(f'results/global_perf.csv', index=False)
# plotGlobalMetricsInOnePlot(df, models_to_plot, metric_to_plot, show=False)

# df1 = getGlobalCorrectedModelResults("bert-based")
# df2 = getGlobalCorrectedModelResults("Bio-bert-based")
# df3 = getGlobalCorrectedModelResults("spark-nlp")
# df = pd.concat([df1, df2, df3], ignore_index=True).drop_duplicates()
# df.to_csv(f'results/global_corrected_perf.csv', index=False)
# Example usage:
# plotGlobalMetricsInOnePlot(df, models_to_plot, metric_to_plot, suffixe="corrected", show=False)