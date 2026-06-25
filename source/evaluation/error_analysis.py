"""
File for preducing all data for error analysis. Given prediction results outputs files with wrong preictions and confusion matrices. 

Arguments: 
[1] path to folder with prediciton files 
"""

# pyright: reportMissingImports=false, reportMissingModuleSource=false
import os, json, sys 
import matplotlib.pyplot as plt
import full_eval 
import pandas as pd 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(font_scale=1.5)
from compute_metrics import is_prediction_file

DATA_DIR = "data"

label_order = {
    13: 1,
    100: 6,
    -1: 3,
    0: 4,
    1:5
}

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

def main(dir):
    """
    Given prediction files in {dir} results computes and outputs information for error analysis. 
    """
    ## get gold labels 
    with open(os.path.join(dir, 'gold.json'), 'r') as f:
        gold = json.load(f)

    #prepare output dir 
    os.makedirs(os.path.join(dir, 'results', 'error_analysis'), exist_ok=True)

    ## get predictions 
    prediction_file = [f for f in os.listdir(dir) if is_prediction_file(f)]
    if len(prediction_file) != 1: raise ValueError(f"To many/not enough files found: {prediction_file}")
    else: prediction_file = prediction_file[0] 

    model_name = os.path.splitext(prediction_file)[0].split('@')[0]
    print(prediction_file)

    with open(os.path.join(dir, prediction_file), 'r') as f:
        predictions = json.load(f)

    for cat in predictions:
        selected_prompt = full_eval.model_prompts[model_name][0] #! we use the model prompt 

        y_pred = predictions[cat][selected_prompt]
        y_true = gold[cat]

        y_pred_sorted = {}
        for k in y_true.keys():
            y_pred_sorted[k] = y_pred[k]
        y_pred = y_pred_sorted
        assert list(y_true.keys()) == list(y_pred.keys())

        # extract reasoning 
        try:
            raw_output = pd.read_csv(os.path.join(dir, f"raw_output_{model_name.split('@')[0]}#test_cat{cat}.csv"), index_col=0)
        except:
            raise LookupError("No raw output file found, or results not from test set")
        
        reasoning = raw_output[selected_prompt]
        reasoning.index = reasoning.index.astype("str")
        assert list(y_true.keys()) == list(reasoning.index)

        get_wrong(model_name, cat, y_true, y_pred, reasoning) #collect alll wrong predictions

        get_cm(model_name, cat, y_true, y_pred) #get confusion matrices 

def get_wrong(model_name, cat, y_true, y_pred, reasoning):
    """
    Collect all wrong predictions of a model on a specific category 

    Paramaters
    model_name (str): name of the LLM
    cat (int): category for which the predictions are given 
    y_true (pd.Series): true labels with NoteID as index 
    y_pred (pd.Series): predicted labels with NoteID as index 
    reasoning (pd.Series): reasoning strings accompanying the prediction with NoteID as index 
    """
    res = {}
    true_false = {}

    for noteID in y_true:
        true_false[noteID] = y_true[noteID] == y_pred[noteID] #whether the prediction is the same as the true label 

    res['correct'] = true_false
    res['gold'] = {k:i2l[v] for k,v in y_true.items()}
    res[model_name] = {k:i2l[v] for k,v in y_pred.items()}

    ## Get notes of wrong predictions
    try: 
        test_path = os.path.join(DATA_DIR, 'dataset', 'new_test_set.csv')
        test = pd.read_csv(test_path, index_col=0)
        test = test[~test[f'cat_{cat}'].isnull()] #remove rows with a None label 
        test.index = test.index.astype("str")

        assert list(test.index) == list(y_true.keys())
        
        res['note'] = test['text']
    except:
        print("Notes not in train set")

    res["reasoning"] = reasoning

    res = pd.DataFrame(res)

    res = res[res[f'correct'] == False] #only keep row where the prediction was wrong 

    res.to_csv(os.path.join(dir, 'results', 'error_analysis', f'wrong_cat{cat}.csv'))

def get_cm(model_name, cat, y_true, y_pred):
    """
    Get the confusion matric of a model's predictions on a specific category 

    Paramaters
    model_name (str): name of the LLM
    cat (int): category for which the predictions are given 
    y_true (pd.Series): true labels with NoteID as index 
    y_pred (pd.Series): predicted labels with NoteID as index 
    """

    # Collect all labels present in either true or predicted 
    present_labels = list(set(list(y_true.values())).union(set(list(y_pred.values()))))
    present_labels.sort(key= lambda val: label_order[val])
    label_names = [i2l[i] for i in present_labels]
    
    cm = confusion_matrix(list(y_true.values()), list(y_pred.values()), labels=present_labels)

    fig, ax = plt.subplots(figsize=(5,5))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="RdPu",
        vmax=20,
        cbar=False,
        square=True,
        xticklabels=label_names,
        yticklabels=label_names
    )
    
    ax.set_xlabel(f"Prediction")
    ax.set_ylabel("True label")
    ax.set_title(i2c[cat])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)  

    fig.subplots_adjust(
    left=0.3,
    right=0.98,
    bottom=0.3,
    top=0.90
    )

    plt.savefig(os.path.join(dir, 'results', 'error_analysis', f'cm_{model_name}_cat{cat}.pdf'))
    plt.close()

if __name__ == "__main__":
    dir = sys.argv[1]
    main(dir)