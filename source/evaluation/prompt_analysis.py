"""
File to analyse the influence of different prompt elements 

Outputs different regression trees based on model performance with different prompts 
"""

# pyright: reportMissingImports=false, reportMissingModuleSource=false
import sys, os
import pandas as pd 
from sklearn import tree
import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from matplotlib import pyplot as plt
import graphviz 
import re

metric = 'f1_macro'

def main(dir):
    """
    Given prediction files in {dir} results create regression trees and collect feature importance 
    """
    feat_imp = {}
    for folder in os.listdir(dir):
        if not os.path.isdir(os.path.join(dir, folder)):
            continue 
        print(os.path.join(dir, folder, "results"))

        folder_dir = os.path.join(dir, folder, "results")
        processed_metrics = collect_metrics(folder_dir)
        feat_scores = get_tree(processed_metrics, folder_dir)

        feat_imp[folder] = feat_scores #collect feature importance scores 

    feat_imp = pd.DataFrame(feat_imp).T 

    feat_imp.loc['mean'] = feat_imp.mean(axis=0)

    feat_imp = feat_imp.round(3)

    feat_imp.to_csv(os.path.join(dir, 'feature_importance.csv'))

def collect_metrics(dir):
    """
    Collect the {metric} of all categories per combination of prompt elements 
    """
    all_metrics = {} 
    for file in os.listdir(dir):
        file_name, ext = os.path.splitext(file)
        # print(file_name)
        if ext != '.csv': #only use metric csv files 
            continue 
        cat = file_name.split("cat")[-1]
        results = pd.read_csv(os.path.join(dir, file), index_col=0, header=0)
        all_metrics[cat] = results.loc[metric] #only get the specified metric scores 
    
    all_metrics = pd.DataFrame(all_metrics)

    processed_metrics = pre_process(all_metrics) #process the row to contain all relevant information 
    return processed_metrics

def _map_range(col):
    """
    maps a series to a range between 0-1 such that the old max is now 1 

    Paramaters:
    col (pd.Series): series of numbers 
    """
    ## shift s.t. max = 1
    try: 
        max = col.max()
        diff = 1 - max 

        return (col + diff) 
    except: #when it's not a series of numbers return original values 
        return col 
    
def pre_process(data: pd.DataFrame):
    """
    Pre-process the data to be suitable for regression tree on prompt elements and metric 
    """

    data = data.apply(_map_range) #shift range 

    data['prompt'] = data.index.str.replace(r'[A-Z]*:', '', regex=True) #create prompt column with only prompt element identifiers
    data = data.melt(id_vars='prompt').drop(columns = ['variable']) #long format 

    data[['T', 'R', 'OG', 'PI']] = data['prompt'].str.split('|', expand= True) # split into different prompt elements 
    data = data.drop(columns='prompt')

    data = pd.get_dummies(data, columns=['T', 'R', 'OG', 'PI'], drop_first=True) #one-hot-encode with drop first to avoid splitting on the same thing

    return data 

def get_tree(df, dir):
    """
    Get the regression tree 

    Parameters: 
    df (pd.DataFrame)
    dir (str): output dir 
    """
    X = df.filter(regex=r'[A-Z]*_.*') #all prompt element columns are the features 
    y = df.filter(['value']) #metric scores 

    dt = DecisionTreeRegressor()
    dt.fit(X, y)

    dot = export_graphviz(
        dt, # pyright: ignore[reportArgumentType]
        feature_names=X.columns,
        filled=True,
        rounded=True,
        precision=5,          # controls decimal places
        impurity=True
    )

    feat_scores = dict(zip(dt.feature_names_in_, dt.feature_importances_))

    dot = re.sub("squared_error", "SE", dot)
    dot = re.sub("value", metric, dot)
    dot = dot.replace(
        'digraph Tree {',
        f'digraph Tree {{\nlabel="{dir}";\nlabelloc="t";\nfontsize=20;'
    )

    graph = graphviz.Source(dot)

    output_path = os.path.join(dir, "prompt_DT")
    graph.render(output_path, format="pdf", cleanup=True)

    return feat_scores

if __name__ == "__main__":
    dir = sys.argv[1]
    main(dir)

