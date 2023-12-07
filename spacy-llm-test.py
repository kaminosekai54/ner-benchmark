import os, sys, time
from spacy_llm.util import assemble
# from usefull import reassembleDataset
import pandas as pd
import json
from sklearn.metrics import classification_report, accuracy_score
from usefull import plot_metrics, evalPerf, getColorForDataset
from tqdm import tqdm
from huggingface_hub import login

# modelType = "huggingface"
modelType = "azure"
# modelType = "antropic"
datasetName = "ncbi-disease"
modelName="azure"
# modelName="llama"
def setupModel(modelName, modelType, datasetName):
    # set your API key as environmental variable :
    apiKey = ""
    with open(f'api-keys/{modelType}-api-key.txt', "r") as keyFile:
        apiKey = keyFile.readlines()[0].strip().replace("\n","")
    if modelType == "open-ai": os.environ["OPENAI_API_KEY"] = apiKey  
    # For Cohere:
    elif modelType == "cohere" : os.environ["CO_API_KEY"] = apiKey 
    #  For Anthropic:
    elif modelType == "antropic" : os.environ["ANTHROPIC_API_KEY"] = apiKey 

    # For PaLM:
    elif modelType == "palm" : os.environ["PALM_API_KEY"] = apiKey 

    #  for azure
    elif modelType == "azure": os.environ["AZURE_OPENAI_KEY"] = apiKey
    
    # for hugginface model :
    elif modelType == "huggingface" : login(token=apiKey)

# modelName="claude2"

    # load the model settings from the config file
    # nlp = assemble(f'spacy-configs/{modelType}/{modelName}_{datasetName}_fewshot.cfg')
    # return nlp

def evalLLM(modelName, modelType, datasetName):
    setupModel(modelName, modelType, datasetName)
    nlp = assemble(f'spacy-configs/{modelType}/{modelName}_{datasetName}_fewshot.cfg')
    
    start_time = time.time()
    limit = 999999

    testDf = pd.read_csv(f"datasets/{datasetName}/test_sample10perc.csv")
    sampledDf = pd.read_csv(f"datasets/{datasetName}/{datasetName}_assembled/test_assembled_sample10perc.csv")
    sampledDf.text = sampledDf.text.apply(lambda x: str(x))
    resultData = []

    # Use tqdm to create a loading bar
    with tqdm(total=len(sampledDf)) as pbar:
        for i, (index, row) in enumerate(sampledDf.iterrows()):
            doc = nlp(row.text)
            # print(doc)
            currentResult = []
            llmAnswer = [(str(ent.text), str(ent.label_)) for ent in doc.ents]
            # print(llmAnswer )
            for word, label in zip(row.text.split(), row.labels.split()):
                if not any(word in foundEntities for foundEntities, _ in llmAnswer):
                    currentResult.append((word, "O"))
                else:
                    for foundEntities, entityLabel in llmAnswer:
                        if word in foundEntities:
                            currentResult.append((word, entityLabel))
                            break
            resultData += currentResult
            pbar.update(1)  # Update the loading bar
            if i >= limit:break

    resultDf = pd.DataFrame(resultData, columns=["text", "labels"])
    resultDf.to_csv(f'results/{datasetName}/{datasetName}_{modelName }_predictions.csv', sep=",", index=False)
    
    resultDf.labels = resultDf.labels.str.lower()
    testDf.labels = testDf.labels.str.lower()

    custom_labels = list(set(list(testDf.labels.unique()) + list(resultDf.labels.unique())))
    for label in testDf.labels.unique():
        if label not in resultDf.labels.unique():
            print(f"warning the label {label} have never been identified")

    report = classification_report(testDf.labels, resultDf.labels, labels=custom_labels,
                                   target_names=custom_labels, output_dict=True)
    data = {
        "modelName": [modelName],
        "dataset": [datasetName],
        "accuracy_global": [round(report['weighted avg']['precision'] * 100, 2)],
        "recall_global": [round(report['weighted avg']['recall'] * 100, 2)],
        "f1_score_global": [round(report['weighted avg']['f1-score'] * 100, 2)]
    }
    for label in custom_labels:
        data[f"accuracy_{label}"] = [round(report[label]['precision'] * 100, 2)]
        data[f"recall_{label}"] = [round(report[label]['recall'] * 100, 2)]
        data[f"f1_score_{label}"] = [round(report[label]['f1-score'] * 100, 2)]

    evalDf = pd.DataFrame(data)
    evalDf.to_csv(f'results/{datasetName}/{datasetName}_{modelName}_eval_corrected.csv', sep=",", index=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("API request time", "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds)))
    return evalDf

metricList = ["accuracy", "recall", "f1_score"]
    # for dataset in os.listdir("datasets/"):
        # print(dataset)
df = evalLLM(modelName=modelName, modelType=modelType, datasetName=datasetName)
for metric in metricList:plot_metrics(df, model_name=modelName, dataset_name=datasetName, metric=metric, show = False, sufixe="corrected", color=getColorForDataset(datasetName))
        
    