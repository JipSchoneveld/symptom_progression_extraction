"""
Get the best performning prompts per category 

Arguments: 
[1]: directory with model predictions and a 'results' folder 
"""

# pyright: reportMissingImports=false, reportMissingModuleSource=false
import pandas as pd 
import json, os, sys, re
from collections import defaultdict

def get_top_prompts(dir, metric): 
    """
    Get the best performing prompts for a results in [dir] given [metric] and collect this in [top_category_prompts]
    """
    model = os.path.basename(dir)
    path = os.path.join(dir, "results")
    res = defaultdict(dict)
    top_category_prompts = {}

    for file in os.listdir(path):
        if not file.lower().endswith(".csv"):
            continue

        if cat := re.search(r'cat([0-9]+)', file): #when there are multiple prompts per category 
            cat = cat.group(1)
        else: #when only one prompt was run, there is no top category prompt 
            return top_category_prompts
        
        df = pd.read_csv(os.path.join(path, file), header=0, index_col=0)
        scores = df.loc[metric]

        if len(scores) < 15:
            return top_category_prompts #only get best if more than 15 prompts  
        
        sorted_scores = scores.sort_values(ascending=False)
        top_prompts = sorted_scores.index[:3]
        res[cat]["prompt_id"] = list(top_prompts)[0]
        res[cat]["top-3"] = list(top_prompts)
        res[cat][metric] = max(scores) 

    res = dict(sorted(res.items()))

    with open(os.path.join(path, "top_prompt_scores.json"), 'w') as f: 
        json.dump(res, f, indent=4)

    top_category_prompts[model] = {str(cat):v['top-3'] for cat, v in res.items()}

    if top_category_prompts: 
        with open(os.path.join(path, 'top_category_prompts.json'), 'w') as f:
            json.dump(top_category_prompts, f, indent=4)

    return top_category_prompts

if __name__ == "__main__":
    dir = sys.argv[1]
    top_prompts = get_top_prompts(dir, "f1_macro")
    print(top_prompts)