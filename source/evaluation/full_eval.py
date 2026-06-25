"""
Perform full evaluation of a folder of prediction folders and collect summary evaluation of the full folder 

Arguments: 
[1]: path of a folder with different folders each containing predictions files of different models 
"""
# pyright: reportMissingImports=false, reportMissingModuleSource=false
import sys, os, json, shutil, re
import compute_metrics
from compute_metrics import is_prediction_file
import pandas as pd 
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
from source.classification.prompt_config import *

metrics = ["f1_macro", "none", "precision_macro"]

i2c = {
  "0": "Depressed mood",
  "1": "Self depreciation and feelings of guilt",
  "2": "Suicidal tendency",
  "3": "Insomnia",
  "4": "Interests and activities",
  "5": "Retardation",
  "6": "Agitation",
  "7": "Anxiety (psychological)",
  "8": "Somatic",
  "9": "Hypochondria",
  "10": "Lack of insight",
  "11": "Weight"
}

def mean_scores(x):
    """
    Get the mean of a list of values 
    """
    # Only process lists
    if isinstance(x, list):
        # If list has 1 item -> return that item
        if len(x) == 1:
            return x[0]
        
        try: #only works if all numeric 
            return np.mean(x)
        except:
            return x
    return x 

def sd_scores(x):
    """
    Get the standard deviation of a list of values 
    """
    # Only process lists
    if isinstance(x, list):
        # If list has 1 item -> return 0
        if len(x) == 1:
            return 0

        try: #only works if all numeric 
            return np.std(x)
        except:
            return x
    return x 

def majority_vote(x):
    """
    Get the most common item in a list 
    """
    x = [e for e in x if e != 13] #do not count None answers 
    if len(x) < 1: #all were 13, so that should be the output 
        return 13
    majority_vote = Counter(x).most_common(1) #when equal now returns random 
    return majority_vote[0][0]

def collect_metric(collected_res, top_category_res, top_model_res, path, metric):   
    """
    Collect all relevant metrics for the summary evaluation

    Parameters:
    collected_res (defaultdict(lambda: defaultdict(list))): dictionary for metrics across all prompts 
    top_category_res (defaultdict(dict)): dictionary for score of top category-prompt 
    top_model_res (defaultdict(dict)): dictionary for score of top model-prompt 
    path (str): path to folder with prediction files
    metric (str): metric to collect 
    """     
    results = os.path.join(path, "results")  
    folder = os.path.basename(path)

    for file in os.listdir(results): 
        _, ext = os.path.splitext(file)
        if ext != '.csv': #only use metric csv files 
            continue 
        scores = pd.read_csv(os.path.join(results, file), index_col=0)

        if cat := re.search(r'cat([0-9]+)', file): #multiple prompts were used 
            cat = cat.group(1)

            #collected scores 
            for _, v in scores.loc[metric].items(): #each row contains the score per prompt id 
                collected_res[f"{folder}"][cat].append(v) 

            #top model and category prompt scores (if present)
            top_category_prompt = category_prompts[folder][cat][0]
            if top_category_prompt in scores: 
                top_category_res[f"{folder}"][cat] = scores.loc[metric, top_category_prompt] 
                top_category_res[f"{folder}_id"][cat] = top_category_prompt

            if folder in model_prompts:
                top_model_prompt = model_prompts[folder][0]
                if top_model_prompt in scores:
                    top_model_res[f"{folder}"][cat] = scores.loc[metric, top_model_prompt]
                    top_model_res[f"{folder}_id"][cat] = top_model_prompt
        else:
            for cat, v in scores.loc[metric].items(): #each row contains the score per cat  
                collected_res[f"{folder}"][cat].append(v) 

    return collected_res, top_category_res, top_model_res

def main(dir, metrics):
    """
    Given a [dir] with different llm-prediction output dirs, compute and collect all summary evaluations in for each metric in [metrics]
    """

    collected_res = {metric: defaultdict(lambda: defaultdict(list)) for metric in metrics}
    top_model_res = {metric: defaultdict(dict) for metric in metrics}
    top_category_res = {metric: defaultdict(dict) for metric in metrics}

    for folder in tqdm(os.listdir(dir), desc="Folders"):
        path = os.path.join(dir, folder)
        if not os.path.isdir(path):
            continue
        tqdm.write(f"Processing {folder}")
        
        #compute individual model/folder results and chack top_prompts info is correct 
        if top_prompts := compute_metrics.analyse_output(path, return_top_prompts=True): #empty if not enough diff prompts 
            if top_prompts[folder] != category_prompts[folder]:
                tqdm.write(f"Top prompts for (last file in) {folder} are not the same as specified in file")

        for metric in metrics: 
            collected_res[metric], top_category_res[metric], top_model_res[metric] = \
                collect_metric(collected_res[metric], top_category_res[metric], top_model_res[metric], path, metric)  

    # get summary statistics and write files
    for metric in metrics: 
        collected_df = pd.DataFrame(collected_res[metric])
        collected_df = collected_df.sort_index()
        collected_df.index = collected_df.index.map(i2c)
        for name, summary_metric in [("sd", sd_scores), ("mean", mean_scores)]: #summarise a range of scores as mean and sd 
            res_df = collected_df.map(summary_metric)
            res_df.loc['Mean'] = res_df.mean(numeric_only=True)
            res_df = res_df.round(3)
            res_df.sort_index(axis=1,inplace=True, key=lambda x: [(col.replace('_id', ''), not col.endswith('_id')) for col in x]) #sort with id column before score column 
            if not all((res_df == 0).all()) or name != "sd": res_df.to_csv(os.path.join(dir, f"{name}_final_eval_{metric}.csv"))

        for name, top_df in [("model", top_model_res[metric]), ("category", top_category_res[metric])]: #collect top model- and category-prompt score 
            top_df = pd.DataFrame(top_df)
            top_df = top_df.map(mean_scores) #if just one value, nothing happens 
            if not top_df.empty:
                top_df = top_df.sort_index()
                top_df.index = top_df.index.map(i2c)
                top_df.loc['Mean'] = top_df.mean(numeric_only=True)
                top_df = top_df.round(3)
                top_df.sort_index(axis=1, inplace=True, key=lambda x: [(col.replace('_id', ''), not col.endswith('_id')) for col in x])
                top_df.to_csv(os.path.join(dir, f"top_{name}_final_eval_{metric}.csv"))

if __name__ == "__main__":
    dir = sys.argv[1]
    main(dir, metrics)