import argparse
import logging
import os
import json
import boto3
import subprocess
import sys
from urllib.parse import urlparse

os.system('pip install autogluon==0.1 seaborn matplotlib')
from autogluon.tabular import TabularDataset, TabularPredictor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import collections
from sklearn.model_selection import StratifiedKFold, train_test_split, ShuffleSplit
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, roc_curve, average_precision_score,
                            precision_recall_curve, precision_score, recall_score, f1_score, matthews_corrcoef, auc)

sns.set(style="whitegrid")

logging.basicConfig(level=logging.DEBUG)
logging.info(subprocess.call('ls -lR /opt/ml/input'.split()))

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train_dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--filename', type=str, default='train.csv')
    parser.add_argument('--target', type=str, default='churn_yn')
    parser.add_argument('--eval_metric', type=str, default='f1')
    parser.add_argument('--presets', type=str, default='good_quality_faster_inference_only_refit')    
    parser.add_argument('--training_minutes', type=str, default=5)    
    parser.add_argument('--debug', type=str, default=False)    
    parser.add_argument('--current_host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--s3_output', type=str, default='s3://autogluon-test/results')
    parser.add_argument('--training_job_name', type=str, default=json.loads(os.environ['SM_TRAINING_ENV'])['job_name'])

    return parser.parse_args()

    
def plot_roc_curve(y_true, y_score, is_single_fig=False):
    """
    Plot ROC Curve and show AUROC score
    """    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.title('AUROC = {:.4f}'.format(roc_auc))
    plt.plot(fpr, tpr, 'b')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.ylabel('TPR(True Positive Rate)')
    plt.xlabel('FPR(False Positive Rate)')
    if is_single_fig:
        plt.show()
    
def plot_pr_curve(y_true, y_score, is_single_fig=False):
    """
    Plot Precision Recall Curve and show AUPRC score
    """
    prec, rec, thresh = precision_recall_curve(y_true, y_score)
    avg_prec = average_precision_score(y_true, y_score)
    plt.title('AUPRC = {:.4f}'.format(avg_prec))
    plt.step(rec, prec, color='b', alpha=0.2, where='post')
    plt.fill_between(rec, prec, step='post', alpha=0.2, color='b')
    plt.plot(rec, prec, 'b')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    if is_single_fig:
        plt.show()

def plot_conf_mtx(y_true, y_score, thresh=0.5, class_labels=['0','1'], is_single_fig=False):
    """
    Plot Confusion matrix
    """    
    y_pred = np.where(y_score >= thresh, 1, 0)
    print("confusion matrix (cutoff={})".format(thresh))
    print(classification_report(y_true, y_pred, target_names=class_labels))
    conf_mtx = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_mtx, xticklabels=class_labels, yticklabels=class_labels, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    if is_single_fig:
        plt.show()


# ------------------------------------------------------------ #
# Hosting methods                                              #
# ------------------------------------------------------------ #

def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.
    :param: model_dir The directory where model files are stored.
    :return: a model (in this case an AutoGluon network)
    """
    net = TabularPredictor.load(model_dir)
    return net


def transform_fn(net, data, input_content_type, output_content_type):
    """
    Transform a request using the Gluon model. Called once per request.
    :param net: The AutoGluon model.
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    :return: response payload and content type.
    """
    data = json.loads(data)
    df_parsed = pd.DataFrame(data)
    prediction = net.predict(df_parsed)
    response_body = json.dumps(prediction.tolist())
    return response_body, output_content_type


def train(args):
    
    # SageMaker passes num_cpus, num_gpus and other args we can use to tailor training to
    # the current container environment, but here we just use simple cpu context.

    model_dir = args.model_dir    
    train_dir = args.train_dir
    filename = args.filename
    target = args.target    
    debug = args.debug
    eval_metric = args.eval_metric   
    presets = args.presets    
    
    num_gpus = int(os.environ['SM_NUM_GPUS'])
    current_host = args.current_host
    hosts = args.hosts
    time_limit = int(args.training_minutes) * 60
     
    logging.info(train_dir)
    
    train_data = TabularDataset(os.path.join(train_dir, filename))
    if debug:
        subsample_size = 500  # subsample subset of data for faster demo, try setting this to much larger values
        train_data = train_data.sample(n=subsample_size, random_state=0)
        
    predictor = TabularPredictor(label=target, path=model_dir, eval_metric=eval_metric).fit(
        train_data=train_data,
        excluded_model_types=['KNN','RF','NN'],
        time_limit=time_limit, presets=[presets, 'optimize_for_deployment'])

    return predictor


def evaluate(predictor, args):
    
    train_dir = args.train_dir
    train_file = args.filename
    test_file = train_file.replace('train', 'test', 1)
    target = args.target
    training_job_name = args.training_job_name
    s3_output = args.s3_output
    presets = args.presets 

    dataset_name = train_file.split('_')[0]
    logging.info(dataset_name)
    
    test_data = TabularDataset(os.path.join(train_dir, test_file))   
    
    u = urlparse(s3_output, allow_fragments=False)
    bucket = u.netloc
    logging.info(bucket)
    prefix = u.path.strip('/')
    logging.info(prefix)
    s3 = boto3.client('s3')
    
    y_test = test_data[target]
    test_data_nolab = test_data.drop(labels=[target], axis=1)
    
    y_pred = predictor.predict(test_data_nolab)
    y_pred_df = pd.DataFrame.from_dict({'True': y_test, 'Predicted': y_pred})
    pred_file = f'{dataset_name}_test_predictions.csv'
    y_pred_df.to_csv(pred_file, index=False, header=True)

    leaderboard = predictor.leaderboard()
    lead_file = f'{dataset_name}_leaderboard.csv'
    leaderboard.to_csv(lead_file)
    
    perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
    #del perf['confusion_matrix']
    perf_file = f'{dataset_name}_model_performance.txt'
    with open(perf_file, 'w') as f:
        print(json.dumps(perf, indent=4, default=pd.DataFrame.to_json), file=f)

    summary = predictor.fit_summary()
    summ_file = f'{dataset_name}_fit_summary.txt'
    with open(summ_file, 'w') as f:
        print(summary, file=f)
    
    y_prob = predictor.predict_proba(test_data_nolab)
    y_prob = y_prob.iloc[:,-1]    
    
    y_test_enc, uniques = pd.factorize(y_test)  # Label Encoding  
            
    fig = plt.figure(figsize=(14,4))
    plt.subplot(1,3,1)
    plot_roc_curve(y_test_enc, y_prob)
    plt.subplot(1,3,2)    
    plot_pr_curve(y_test_enc, y_prob)
    plt.subplot(1,3,3)    
    plot_conf_mtx(y_test_enc, y_prob, 0.5) 
    eval_file = f'{dataset_name}_eval.png'
    plt.savefig(eval_file)
    plt.close(fig)

#     # Feature importance
#     featimp = predictor.feature_importance(test_data)
#     fig, ax = plt.subplots(figsize=(12,5))
#     plot = sns.barplot(x=featimp.index, y=featimp.values)
#     ax.set_title('Feature Importance')
#     plot.set_xticklabels(plot.get_xticklabels(), rotation='vertical')
#     featimp_imgfile = f'{dataset_name}_featimp.png'
#     featimp_csvfile = f'{dataset_name}_featimp.csv'
#     fig.savefig(featimp_imgfile)
#     featimp.to_csv(featimp_csvfile)
#     plt.close(fig)        
        
    # Cleanup data in order to avoid disk space issues
    predictor.save_space()
    predictor.delete_models(models_to_keep='best', dry_run=False)

    files_to_upload = [pred_file, lead_file, perf_file, summ_file, eval_file]
    for file in files_to_upload:
        s3.upload_file(file, bucket, os.path.join(prefix, training_job_name.replace('mxnet-training', 'autogluon', 1), file))   

            
if __name__ == '__main__':
    args = parse_args()
    predictor = train(args)
    evaluate(predictor, args)