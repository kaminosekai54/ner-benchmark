################################################################################
#imports
import pandas as pd
import time, os
from usefull import correctLabels
from usefull import plot_metrics
from usefull import evalPerf
from usefull import getColorForDataset

#  loop over dataset and model to generate plots
modelList = ["Bio-bert-based", "bert-based", "spark-nlp"]
metricList = ["accuracy", "recall", "f1_score"]

for dataset in os.listdir("datasets/"):
    print(dataset)
    for model in modelList:
        print(model)
        df = evalPerf(datasetName=dataset, modelName=model)
        dfCorrected = correctLabels(datasetName=dataset, modelName=model)
        for metric in metricList:
            plot_metrics(df, model_name=model, dataset_name=dataset, metric=metric, show = False, sufixe="", color=getColorForDataset(dataset))
            plot_metrics(dfCorrected, model_name=model, dataset_name=dataset, metric=metric, show = False, sufixe="corrected", color=getColorForDataset(dataset))