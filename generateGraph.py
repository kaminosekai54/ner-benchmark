################################################################################
#imports
import pandas as pd
import time, os
from usefull import correctLabels
from usefull import plot_metrics

#  loop over dataset and model to generate plots
modelList = ["Bio-bert-based", "bert-based", "spark-nlp"]
metricList = ["accuracy", "recall", "f1_score"]
for dataset in os.listdir("datasets/"):
    print(dataset)
    for model in modelList:
        print(model)
        df = correctLabels(datasetName=dataset, modelName=model)
        for metric in metricList:
            plot_metrics(df, model_name=model, dataset_name=dataset, metric=metric, show = False, sufixe="corrected")