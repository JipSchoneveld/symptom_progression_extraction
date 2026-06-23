"""
Main file for training and testing for symptom status prediction.

depends on: 
- dataset csv files in specified directory: /data/dataset/[train|test]_set.csv

arguments: 
output_dir (str): #folder where results should be saved 
llm_loc (str): hugging face llm path 
temp (float): temperature 
prompt_mode ("model"|"all"|"category"): use the model model/category prompt(s) or all variations 
prompt_style ("cot"|"sot"|"original"): different prompt styles 
use_test (bool): whether to use the test set or only train

outputs folder in specified {output_dir} with:
- raw LLM output 
- json files of all models that were run 
- json file with the gold labels 
- txt file of used prompts 
- txt file with the used experimental configurations 
"""

# pyright: reportMissingImports=false, reportMissingModuleSource=false
import pandas as pd 
import os, sys, json
import datetime
import time
import numpy as np 
from tqdm import tqdm
from collections import defaultdict 
from vllm import LLM
from source.classification.symptom_baselines import majority_baseline, keyword_baseline
from source.classification.llm_models import make_llm_prediction
from source.classification.prompts import *
from source.classification.prompt_config import *

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0" 

OUTPUT_DIR = "output" 
DATA_DIR = "data"  

## <EXPERIMENT SETTINGS> ##
output_dir = 'Qwen2.5-7B-Instruct' #folder where results should be saved 
llm_loc = 'Qwen/Qwen2.5-7B-Instruct'
temp = 0; prompt_mode = "model"; prompt_style = "original"
use_test = True

def create_dataset(data, cat):
    """
    Selects the relevant labels for the category given the dataset. Splits into x and y while keeping the original indices. 

    Parameters: 
    data (pd.DataFrame): dataset with x values and y values 
    cat (int): category number 

    Returns:
    x (pd.Series): input text
    y (pd.Series): gold output labels 
    """
    filtered_data = data[~data[f'cat_{cat}'].isnull()] #remove rows with a None label 
    x = filtered_data['text']; y = filtered_data[f'cat_{cat}']

    return x, y

def run_baselines(x_train, y_train, cat, x_test = None):
    """
    Run the baseline model(s) on the dataset. If x_test is given, returned predictions are on the test set, otherwise predictions are on the train set. 

    Parameters: 
    x_train (pd.Series): input text from train set with original indices 
    y_train (pd.Series): gold output labels with original indices 
    cat (int): the category for which predictions are made 
    x_test (pd.Series): input text from test set with original indices, default is None 

    Returns:
    res (dict[str, dict[int, dict[int, int]): all predicted labels as a dictionary of: model-type -> category -> NoteID -> predicted label
    """
    res = {
        'majority': {}, 
        'keyword': {}
    }

    mb = majority_baseline(x_train, y_train, x_test)
    res['majority'][cat] = mb

    kw = keyword_baseline(x_train, cat, x_test)
    res['keyword'][cat] = kw

    return res 

def run_llms(x_train, y, cat, x_test = None):
    """
    Run the LLM on the train set or test set if given. NB, when the test is given, predictions are only run on the test set. 

    Parameters: 
    x_train (pd.Series): input text from train set with original indices 
    y_train (pd.Series): gold output labels with original indices 
    cat (int): the category for which predictions are made 
    x_test (pd.Series): input text from test set  with original indices 

    Returns:
    res (dict[str, dict[int, dict[int, int]): all predicted labels as a dictionary of: llm -> category -> NoteID -> predictions 
    """

    res = defaultdict(dict)

    if prompt_style == "cot":
        all_prompts = get_cot_prompts(cat)
    elif prompt_style == "sot":
        all_prompts = get_sot_prompts(cat)
    elif prompt_style == "original":
        all_prompts = get_prompts(cat)
    else: 
        raise ValueError(prompt_style, "is not a valid promptstyle.")
    
    if prompt_mode == "all":
        prompts = all_prompts
        tqdm.write(f"Running all {len(prompts)} prompts")
    elif prompt_mode == "model":
        ids = model_prompts[llm_loc]
        if temp != 0 or prompt_style != "original":   #when using T>0 or prompt modifier technique, run only the best prompt, otherwise top-3
            ids = [ids[0]]
        prompts = {id: all_prompts[id] for id in ids}
    elif prompt_mode == "category":
        ids = category_prompts[llm_loc][str(cat)]
        if temp != 0 or prompt_style != "original":   #when using T>0 or prompt modifier technique, run only the best prompt, otherwise top-3 
            ids = [ids[0]]
        prompts = {id: all_prompts[id] for id in ids}
        tqdm.write(f"Running {len(prompts)} category prompt(s)")
    else: 
        raise ValueError(f"{prompt_mode} is not a valid value for [prompt_mode], use 'all', 'model', or 'category'")
    
    if temp == 0: seeds = [42]
    else: seeds = [91385, 25641, 87751, 71710, 1546] #randomly selected seeds 
    
    #run the experiments for each seed 
    for seed in tqdm(seeds, desc = "model runs", position = 1, file = sys.stdout, leave = False, dynamic_ncols=True): 
        model_name = f"{llm_loc.split('/')[1]}@{seed}"
        start = time.time()
        res[model_name][cat] = make_llm_prediction(loaded_model, llm_loc, prompts, x_train, y, cat, output_dir, temp = temp, seed = seed, x_test = x_test)
        with open(os.path.join(OUTPUT_DIR, output_dir, 'config.txt'), 'a') as f: 
            f.write(f"\n{model_name} for cat_{cat}: finished at: {datetime.datetime.now()} | total time: {int(time.time() - start)} seconds")
    return res 

def main():
    """
    Given the datasets at {DATA_DIR}/dataset/(train|test)_set.csv
    -> train and run various models and save their outputs on the test set at {OUTPUT_DIR}/{output_dir}. 

    """

    #setup dictionary for collecting results 
    models = defaultdict(dict)

    # setup output folder
    os.makedirs(os.path.join(OUTPUT_DIR, output_dir), exist_ok=True)
    
    # set train set
    train_path = os.path.join(DATA_DIR, 'dataset', 'llm_val_set.csv')
    train = pd.read_csv(train_path, index_col=0)
    # train = train.head(3)
    print(f"Train set:\n{train}")

    if use_test:
        test = pd.read_csv(os.path.join(DATA_DIR, 'dataset', 'test_set.csv'), index_col=0)
    else:
        test = None 
    print(f"Test set:\n{test}")

    #save configurations
    with open(os.path.join(OUTPUT_DIR, output_dir, 'config.txt'), 'w') as f: 
        f.write(f"{datetime.datetime.now()}\n")
        f.write(f"LLM: {llm_loc}\n")
        f.write(f"temp = {temp}; prompt_mode = {prompt_mode}; prompt_style = {prompt_style}")
        f.write(f"Train: {train_path} - {len(train)}\n")
        f.write(f"Test: {len(test) if test is not None else '-'}")

    # run everything for each category
    for cat in tqdm([0, 2, 3, 4, 7, 8], desc = "symptom categories", position = 0, file = sys.stdout, dynamic_ncols=True):

        x_train, y_train = create_dataset(data = train, cat = cat) 
        if test is not None:
            x_test, y_test = create_dataset(data = test, cat = cat) 
        else: #don't use a test set 
            x_test, y_test = None, None

        ## save gold labels
        if y_test is not None:
            models['gold'][cat] = {id:label for id, label in y_test.items()} 
        else: 
            models['gold'][cat] = {id:label for id, label in y_train.items()}

        ## LLMs
        for model, results in run_llms(x_train, y_train, cat, x_test).items():
            models[model].update(results)

        ## baselines
        for model, results in run_baselines(x_train, y_train, cat, x_test).items():
            models[model].update(results)


    #save the output of each model 
    for model, results in models.items():
        with open(os.path.join(OUTPUT_DIR, output_dir, f'{model}.json'), 'w') as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    if llm_loc is not None:
        print(f"Loading {llm_loc}", end='\n\n')
        loaded_model = LLM(
            model = llm_loc, 
            seed = 42, 
            enable_prefix_caching=False, 
            gpu_memory_utilization = 0.9, 
            # max_model_len = 2560, #only for qwen2.5-32B
            # max_model_len = 40960, #only for gemma
            enforce_eager = True, 
            max_num_seqs = 256 #Default: 256
        )
        print("Done!", end = '\n\n')
    main()
