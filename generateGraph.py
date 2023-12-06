################################################################################
#imports
import pandas as pd
import time, os
from usefull import correctLabels, plot_metrics, evalPerf, getColorForDataset, generateGlobalMetricsPlot

#  loop over dataset and model to generate plots
modelList = ["Bio-bert-based", "bert-based", "spark-nlp"]
metricList = ["f1_score", "accuracy", "recall"]

for dataset in os.listdir("datasets/"):
    print(dataset)
    for model in modelList:
        print(model)
        df = evalPerf(datasetName=dataset, modelName=model)
        dfCorrected = correctLabels(datasetName=dataset, modelName=model)
        for metric in metricList:
            plot_metrics(df, model_name=model, dataset_name=dataset, metric=metric, show = False, sufixe="", color=getColorForDataset(dataset))
            plot_metrics(dfCorrected, model_name=model, dataset_name=dataset, metric=metric, show = False, sufixe="corrected", color=getColorForDataset(dataset))
            
generateGlobalMetricsPlot(modelList=modelList, metricList=[f"{metric}_global" for metric in metricList])