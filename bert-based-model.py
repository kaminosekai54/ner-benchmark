## inspired by :
#https://colab.research.google.com/github/eugenesiow/practical-ml/blob/master/notebooks/Named_Entity_Recognition_BC5CDR.ipynb

################################################################################
#imports
import pandas as pd
from simpletransformers.ner import NERModel
from transformers import AutoTokenizer
import logging
import time, os
from usefull import readDatasetWithSentenceId
from usefull import plot_metrics
from sklearn.metrics import classification_report, accuracy_score


################################################################################
# functions
def evalBertBasedModel(datasetName, modelName = "bert-based"):
    if not os.path.isdir("./results/"): os.makedirs("./results/")
    if not os.path.isdir(f'results/{datasetName}/'): os.makedirs(f'results/{datasetName}/')
    
    train_df = readDatasetWithSentenceId(datasetName, "train")
    test_df = readDatasetWithSentenceId(datasetName, "test")
    dev_df = readDatasetWithSentenceId(datasetName, "dev")
    
    # Training and Testing the Model
    #Set up the Training Arguments
    train_args = {
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    'sliding_window': True,
    'max_seq_length': 64,
    'num_train_epochs': 10,
    'train_batch_size': 24,
    'fp16': True,
    'use_early_stopping': True,
    'save_model_every_epoch': False,
    'output_dir': f'./outputs/{modelName}/',
    'best_model_dir': f'/outputs/{modelName}/best_model/',
    'evaluate_during_training': True,
    }

    # save a list of all unique label in training data set
    custom_labels = list(train_df['labels'].unique())

    #Train the Model
    # using the pre-trained BioBERT model (by [DMIS Lab, Korea University](https://huggingface.co/dmis-lab)) 
    # log level for more or less verbosity
    # logging.basicConfig(level=logging.Warning)
    transformers_logger = logging.getLogger('transformers')
    transformers_logger.setLevel(logging.WARNING)
    # using the bio BERT pre-trained model.
    model = NERModel('bert', 'bert-base-uncased', labels=custom_labels, args=train_args)

    # Train the model
    start_time = time.time()
    model.train_model(train_df, eval_data=dev_df)
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("training time", "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds)))
    # Evaluate the model in terms of accuracy score
    result, model_outputs, preds_list = model.eval_model(test_df)
    
    # make prediction to be able to have entity level evaluation
    y_pred = model.predict(test_df.words.values.tolist())
    resultList= [(k,v) for d in [x for line in y_pred[0] for x in line] for k,v in d.items()]
    tdf = pd.DataFrame(resultList, columns=["words", "labels"])
    tdf.to_csv(f'results/{datasetName}/{datasetName}_{modelName }_predictions.csv', sep=",", index=False)
    custom_labels += [f'{label}_new' for label in tdf.labels.unique() if label not in custom_labels]
    
    # Calculate precision, recall, and F1-score globally and for each unique label
    report = classification_report(test_df['labels'], tdf.labels, labels=custom_labels, target_names=custom_labels, output_dict=True)

    loss_global = result['eval_loss']
    accuracy_global = result['precision']
    recall_global = result['recall']
    f1_score_global = result['f1_score']

    # Create a data frame with the desired output
    data = {
        "modelName": [modelName ],
        "dataset": [datasetName],
        "loss": [round(loss_global, 5)],
        "accuracy_global": [round(accuracy_global, 2)],
        # "precision_global": [round(precision_global *100, 2)],
        "recall_global": [round(recall_global*100, 2)],
        "f1_score_global": [round(f1_score_global *100, 2)]
    }
    for label in custom_labels:
        data[f"accuracy_{label}"] = [round(report[label]['precision']*100, 2)]
        data[f"recall_{label}"] = [round(report[label]['recall']*100, 2)]
        data[f"f1_score_{label}"] = [round(report[label]['f1-score']*100, 2)]

    df = pd.DataFrame(data)
    df.to_csv(f'results/{datasetName}/{datasetName}_{modelName }_eval.csv', sep=",", index=False)
    return df

if __name__ == '__main__':
    # model_name="Bio-bert-based"
    model_name="bert-based"
    metricList = ["accuracy", "recall", "f1_score"]
    for dataset in os.listdir("datasets/"):
        print(dataset)
        df = evalBertBasedModel(datasetName=dataset)
        for metric in metricList:
            plot_metrics(df, model_name=model_name, dataset_name=dataset, metric=metric, show = False)
            
    #    if you want to do evaluation of a specific dataset, uncomment the line bellow and comment the loop above     # 
    # dataset= "bc5cdr"
    # dataset= "jnlpba"
    # dataset= "ncbi-disease"
    # dataset= "ontonote"
    # dataset= "species800"
    # df = evalBertBasedModel(dataset)
    # for metric in metricList: plot_metrics(df, model_name=model_name, dataset_name=dataset, metric=metric, show = False)