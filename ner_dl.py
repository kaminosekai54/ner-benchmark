from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark import SparkContext as sc
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
import pandas as pd
from collections import defaultdict
from sklearn.metrics import classification_report, accuracy_score
import time, os
import sparknlp
from usefull import readDatasetWithSentenceId
from usefull import plot_metrics
from usefull import getEncoding

################################################################################
# functions
def evalSparkNLPModel(datasetName, modelName = "spark-nlp", startSpark=True, spark=None):
    # start spark session to be able to use it's component
    if startSpark==True : spark = sparknlp.start()
    elif spark is not None: spark  = spark
    else: spark = sparknlp.start()

    # we create results folders if not exist 
    if not os.path.isdir("./results/"): os.makedirs("./results/")
    if not os.path.isdir(f'results/{datasetName}/'): os.makedirs(f'results/{datasetName}/')
# we use the pretrained model from spark-nlp
    ner_model= NerDLModel.pretrained()
    # we prepare the elements for the pipeline
    document = DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")

    sentence = SentenceDetector()\
        .setInputCols(['document'])\
        .setOutputCol('sentence')

    token = Tokenizer()\
        .setInputCols(['sentence'])\
        .setOutputCol('token')

    embeddings = WordEmbeddingsModel.pretrained()\
    .setOutputCol('embeddings')

    prediction_pipeline = Pipeline(
        stages = [
            document,
            sentence,
            token,
            embeddings,
            ner_model
        ]
    )

    # now importing the training and testing data
    train_df= pd.read_csv(f'datasets/{datasetName}/{datasetName}_assembled/train_assembled.tsv', sep="\t", dtype={"text":str, "labels":str}, encoding=getEncoding(f'datasets/{datasetName}/{datasetName}_assembled/train_assembled.tsv'))
    train_df= train_df.assign(textList = train_df.text)
    train_df.textList  = train_df.textList .apply(lambda x : [str(x)])
    prediction_data = spark.createDataFrame(train_df.textList .values.tolist()).toDF("text")
    prediction_model = prediction_pipeline.fit(prediction_data)
    lp = LightPipeline(prediction_model)
    # test_df = pd.read_csv(f'datasets/{datasetName}/test.tsv', sep="\t", names=["text", "labels"], dtype={"text":str, "labels":str}, encoding=getEncoding(f'datasets/{datasetName}/test.tsv'))
    test_df = readDatasetWithSentenceId(datasetName, "test")
    test_df["text"] = test_df.words.apply(lambda x : str(x)) 
    start_time = time.time()
    print("starting evaluation")
    result = lp.annotate(test_df.text.values.tolist())
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("evaluation time", "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds)))
    formatedResults = defaultdict(list)
    for d in result:
        for key, value in d.items():
            formatedResults[key].append(value)

    formatedResults = dict(formatedResults)
    predictedDf = pd.DataFrame.from_dict(formatedResults)
    predictedDf["text"] = predictedDf.sentence.apply(lambda x : str(x[0]))
    tdf= predictedDf[["text", "ner"]]
    tdf["labels"]= tdf.ner.apply(lambda x : str(x[0]))
    tdf.to_csv(f'results/{datasetName}/{datasetName}_{modelName }_predictions.csv', sep=",", index=False)

    custom_labels = list(test_df.labels.unique())
    custom_labels += [f'{label}_new' for label in tdf.labels.unique() if label not in custom_labels]

    # Calculate precision, recall, and F1-score globally and for each unique label
    report = classification_report(test_df.labels, tdf.labels, labels=custom_labels, target_names=custom_labels, output_dict=True)
    # print(report)

    # Create a data frame with the desired output
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
    df.to_csv(f'results/{datasetName}/{datasetName}_{modelName }_eval.csv', sep=",", index=False)
    return df

if __name__ == '__main__':
    spark = sparknlp.start()
    model_name="spark-nlp"
    metricList = ["accuracy", "recall", "f1_score"]
    for dataset in os.listdir("datasets/"):
        print(dataset)
        df = evalSparkNLPModel(datasetName=dataset, startSpark=False, spark=spark)
        for metric in metricList:
            plot_metrics(df, model_name=model_name, dataset_name=dataset, metric=metric, show = False)
            
            
    # dataset= "species800"
    # df = evalSparkNLPModel(dataset)
    # for metric in metricList: plot_metrics(df, model_name=model_name, dataset_name=dataset, metric=metric, show = False)