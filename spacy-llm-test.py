import os, sys, time
from spacy_llm.util import assemble
# from usefull import reassembleDataset
import pandas as pd
import json
from sklearn.metrics import classification_report, accuracy_score
from usefull import plot_metrics, evalPerf, getColorForDataset

# modelType = "huggingface"
modelType = "azure"
# modelType = "antropic"
datasetName = "ncbi-disease"
modelName="azure"
# set your API key as environmental variable :
apiKey = ""
with open(f'api-keys/{modelType}-api-key.txt', "r") as keyFile:
    apiKey = keyFile.readlines()[0].strip().replace("\n","")
# os.environ["OPENAI_API_KEY"] = apiKey  
# For Cohere:
os.environ["CO_API_KEY"] = apiKey 
#  For Anthropic:
os.environ["ANTHROPIC_API_KEY"] = apiKey 

# For PaLM:
os.environ["PALM_API_KEY"] = apiKey 

#  for azure
os.environ["AZURE_OPENAI_KEY"] = apiKey 

# modelName="claude2"

# load the model settings from the config file
nlp = assemble(f'spacy-configs/{modelName}_fewshot.cfg')
# print(dir(nlp))
# doc = nlp("Begin by preparing the flavorful meat sauce. In a large pan over medium heat, sauté finely chopped onions in olive oil until translucent. Add minced garlic and ground beef, cooking until the beef is browned. Break apart the meat with a spoon for an even texture.")
def evalLLM(modelName, datasetName):
    start_time = time.time()
    limit = 999999

    testDf= pd.read_csv(f"datasets/{datasetName}/test_sample10perc.csv")
    sampledDf= pd.read_csv(f"datasets/{datasetName}/{datasetName}_assembled/test_assembled_sample10perc.csv")
    sampledDf.text = sampledDf.text.apply(lambda x :  str(x))
    resultData = []
    for i, (index, row) in enumerate(sampledDf.iterrows()):
        print(i)
        doc = nlp(row.text)
        currentResult = []
        llmAnswer = [(str(ent.text), str(ent.label_)) for ent in doc.ents]
        for word, label in zip(row.text.split(), row.labels.split()):
            if not any(word in foundEntities for foundEntities, _ in llmAnswer):
                currentResult.append((word, "O"))
            else:
            # print([(ent.text, ent.label_) for ent in doc.ents])
                for foundEntities, entityLabel in llmAnswer:
                    if word in foundEntities  :
                        currentResult.append((word, entityLabel))
                        break
        resultData += currentResult
        if i >= limit : break
    
    resultDf= pd.DataFrame(resultData, columns=["text", "labels"])
    resultDf.labels = resultDf.labels.str.lower()
# testDf= testDf.head(len(resultDf))
    testDf.labels = testDf.labels.str.lower()

    custom_labels = list(set(list(testDf.labels.unique()) + list(resultDf.labels.unique())))
    for label in testDf.labels.unique():
        if not label in resultDf.labels.unique():
            print(f"warning the label {label} have never been identified")
        
    report = classification_report(testDf.labels, resultDf.labels, labels=custom_labels, target_names=custom_labels, output_dict=True)
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

    evalDf = pd.DataFrame(data)
    evalDf.to_csv(f'results/{datasetName}/{datasetName}_{modelName }_eval_corrected.csv', sep=",", index=False)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("API request time", "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds)))
    return evalDf
    
    
if __name__ == '__main__':
    metricList = ["accuracy", "recall", "f1_score"]
    for dataset in os.listdir("datasets/"):
        print(dataset)
        df = evalLLM(modelName=modelName, datasetName=dataset)
        for metric in metricList:
            plot_metrics(df, model_name=modelName, dataset_name=dataset, metric=metric, show = False, sufixe="corrected", color=getColorForDataset(dataset))
        
    