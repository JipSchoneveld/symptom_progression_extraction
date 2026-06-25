"""
Main file for analysing the predicted outputs for symptom status prediction

depends on: 
- prediction json files in specified directory {dir}
- list of all possible labels {LABELS}

Required arguments: 
- [1] path to folder with prediction files 

outputs a 'results' folder with:
- standard scores: recall, precision, accuracy, recall_binary, precision_binary, accuracy_binary, recall_(-1,0,1), precision_(-1,0,1)
- F-scores: f1_macro, f1_micro, f1_macro_(-1,0,1), f1_({label}), f1_binary
- ONLY FOR LLMs: none (= number of instances with a non-conforming output)
"""

# pyright: reportMissingImports=false, reportMissingModuleSource=false
import os, json, sys
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import pandas as pd
from collections import defaultdict
from get_top_prompts import get_top_prompts 

LABELS = [-1,0,1,100]

i2l = {
    -1: "worsened",
    0: "established",
    1: "improved",
    100: "no mention",
    13: "none"
}

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

def is_prediction_file(file_name: str): 
    """
    Checks if provided file is prediction file, returns False otherwise 
    """
    res = True
    _, ext = os.path.splitext(file_name)
    if file_name == 'gold.json': #gold labels 
        res = False
    if ext != '.json': #not the correct format
        res = False 
    if 'prompt' in file_name: #not predictions
        res = False

    return res 

def analyse_output(dir, return_top_prompts = False):
    """
    Computes evaluation scores for all available predictions in {dir} and writes them to csv files
    """

    top_prompts = None
    #setup results folder in output dir
    os.makedirs(os.path.join(dir, 'results'), exist_ok=True)

    ## get gold labels 
    with open(os.path.join(dir, 'gold.json'), 'r') as f:
        gold = json.load(f)

    ## get predictions 
    for file_name in os.listdir(dir):
        model_name, _ = os.path.splitext(file_name)
        if not is_prediction_file(file_name): #only use prediction files 
            continue 
        with open(os.path.join(dir, file_name), 'r') as f:
            all_predictions = json.load(f)

        metrics = evaluate(gold, all_predictions)

        write_files(model_name, metrics, dir)

        top_prompts = get_top_prompts(dir, 'f1_macro')
    if return_top_prompts: return top_prompts #nb returns the last one 

def write_files(model_name, metrics, dir):
    """
    Write files containing the prediction metrics {metrics} 
    """
    
    #if any catgeory has prediction metrics for more than 1 prompt, we write a seperate file per category 
    if any([len(prompts) > 1 for c, prompts in metrics.items()]): 
        for cat, predictions in metrics.items():
            predictions = pd.DataFrame(predictions)
            predictions.to_csv(os.path.join(dir, 'results', f'{model_name}_cat{cat}.csv'))
    else:
        #save prompt IDs
        prompt_ids = [
            list(predictions.keys())[0] for cat, predictions in metrics.items() 
        ]
        
        with open(os.path.join(dir, 'results', f'{model_name}_prompt_ids.txt'), "w") as f:
            f.writelines([p +'\n' for p in prompt_ids])
        
        #Save metrics without prompt IDs
        metrics = {
            cat: list(predictions.values())[0] for cat, predictions in metrics.items() #remove prompt id since all categories have 1 prompt ID 
        }

        metrics = pd.DataFrame(metrics)
        metrics.to_csv(os.path.join(dir, 'results', f'{model_name}.csv'))

def evaluate(gold, all_predictions):
    """
    Collects the metrics

    Parameters: 
    gold (dict): the gold labels as a dictionary of: category -> noteID -> label 
    predictions (dict): predicted labels as a dictionary of: category -> noteID -> label 

    Returns: 
    metrics (dict[int, dict[str, float]]): al metrics for all categories as a dictionary of: category -> prompt id -> metric -> score 
    """

    ##setup dictionary
    metrics = defaultdict(lambda: defaultdict(dict))

    ## calculate metrics for all categories 
    for cat, predictions in all_predictions.items():
        y_true = gold[cat]
        for prompt_id, y_pred in predictions.items():

            if type(y_pred) != dict: 
                y_pred = predictions
                prompt_id = 'baseline'

            # make sure the predictions are in the same order as the gold labels 
            y_pred_sorted = {}
            for k in y_true.keys():
                y_pred_sorted[k] = y_pred[k]

            # check that the noteIDs are the same 
            assert list(y_true.keys()) == list(y_pred_sorted.keys()), f"y_true: {y_true.keys()}\ny_pred: {y_pred_sorted.keys()} \
                \nmissing: {set(y_true.keys()) - set(y_pred_sorted.keys())}\nextra: {set(y_pred_sorted.keys()) - set(y_true.keys())}"
            
            # get only the labels  
            y_true_labels = list(y_true.values()); y_pred_sorted = list(y_pred_sorted.values())

            present_labels = list(set(y_true_labels)) #labels that are present in y_true for this category

            none_i = [i for i, elem in enumerate(y_pred_sorted) if elem == 13] #collect indices of None items
            metrics[cat][prompt_id]['none'] = len(none_i) #number of none items 

            # collect all metrics 
            
            metrics[cat][prompt_id].update(compute_standard_scores(y_true_labels, y_pred_sorted))
            metrics[cat][prompt_id].update(compute_f_scores(y_true_labels, y_pred_sorted, present_labels))

            if prompt_id == 'baseline': #you only have to do it once
                break
    
    return dict(metrics)

def compute_standard_scores(y_true, y_pred):
    """
    Compute all standard evluation metrics (recall, precision, accuracy) for normal, binary, and mentioned only settings. 

    Parameters: 
    y_true: list of all gold labels in corresponding order to y_pred
    y_pred: list of all gold labels in corresponding order to y_true

    Returns: 
    res (dict[str, float]): all metrics as a dictionary of: metric -> score 
    """

    ##setup results dictionary
    res = {}

    ## normal metrics 
    res['recall_macro'] = recall_score(y_true=y_true, y_pred=y_pred, average = 'macro', labels=LABELS, zero_division=0) 
    res['precision_macro'] = precision_score(y_true=y_true, y_pred=y_pred, average = 'macro', labels=LABELS, zero_division=0) 
    res['recall_micro'] = recall_score(y_true=y_true, y_pred=y_pred, average = 'micro', labels=LABELS, zero_division=0) 
    res['precision_micro'] = precision_score(y_true=y_true, y_pred=y_pred, average = 'micro', labels=LABELS, zero_division=0) 
    res['accuracy'] = accuracy_score(y_true=y_true, y_pred=y_pred) 

    ## binary metrics (treat 1, 0, and -1 all as 'mention' labels)
    #convert to binary labels 
    y_true_binary = ['no mention' if a == 100 else 'mention' for a in y_true]
    y_pred_binary = ['no mention' if a == 100 else 'mention' for a in y_pred]

    # calculate binary metrics 
    res['recall_binary'] = recall_score(y_true=y_true_binary, y_pred=y_pred_binary, pos_label='mention', labels=LABELS, zero_division=0)
    res['precision_binary'] = precision_score(y_true=y_true_binary, y_pred=y_pred_binary, pos_label='mention', labels=LABELS, zero_division=0)
    res['accuracy_binary'] = accuracy_score(y_true=y_true_binary, y_pred=y_pred_binary)

    ## mention metrics (ignore the no-mention label (100))
    res['recall_macro_(-1,0,1)'] = recall_score(y_true=y_true, y_pred=y_pred, average = 'macro', labels = [-1,0,1], zero_division=0) 
    res['precision_macro_(-1,0,1)'] = precision_score(y_true=y_true, y_pred=y_pred, average = 'macro', labels = [-1,0,1], zero_division=0) 

    return res

def compute_f_scores(y_true: list, y_pred: list, present_labels: list):
    """
    Compute all f-scores (macro, micro, macro-mention, per-class, binary).

    Parameters: 
    y_true: list of all gold labels in corresponding order to y_pred
    y_pred: list of all gold labels in corresponding order to y_true
    present_labels: list of all present labels in y_true

    Returns: 
    res (dict[str, float]): all metrics as a dictionary of: metric -> score 
    """

    ##setup dictionary 
    res = {}

    ## combined macro and micro 
    res['f1_macro'] = f1_score(y_true= y_true, y_pred= y_pred, average = 'macro', labels = LABELS, zero_division=0)  
    res['f1_micro'] = f1_score(y_true= y_true, y_pred= y_pred, average = 'micro', labels = LABELS, zero_division=0) 

    ## mentioned macro-f1 (ignore no-mention label)
    res['f1_macro_(-1,0,1)'] = f1_score(y_true= y_true, y_pred= y_pred, average = 'macro', labels = [-1,0,1], zero_division=0) 

    ## present macro-f1 (ignore labels which are not in y true)
    res['f1_macro_present'] = f1_score(y_true= y_true, y_pred= y_pred, average = 'macro', labels = present_labels, zero_division=0) #take average only of labels present in dataset 

    ## per class f1 (treat as binary per class)
    class_f1 = f1_score(y_true=y_true, y_pred=y_pred, average=None, labels=LABELS, zero_division=0) 
    for i, label in enumerate(LABELS):
        res[f'f1_({label})'] = class_f1[i]  # type: ignore

    ## binary f1 (treat 1, 0, and -1 all as 'mention' labels)
    #convert to binary labels 
    y_true_binary = ['no mention' if a == 100 and a!= 13 else 'mention' for a in y_true]
    y_pred_binary = ['no mention' if a == 100 and a!= 13 else 'mention' for a in y_pred]

    res['f1_binary'] = f1_score(y_true=y_true_binary, y_pred=y_pred_binary, pos_label='mention', zero_division=0) 
    
    return res

if __name__ == "__main__":
    dir = sys.argv[1]
    analyse_output(dir)
