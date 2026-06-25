# Symptom Progression Extraction from Psychiatric Clinical Notes Using LLMs

This repo contains the code for the paper: 

> **Symptom Progression Extraction from Psychiatric Clinical Notes Using LLMs** <br>
_Jip Schoneveld, Gizem Sogancioglu, Heysem Kaya, Karin Hagoort, Stan J. Meyberg, Cas Verhappen, Albert A. Salah, Floortje E. Scheepers_

## Structure                
    .
    ├── source                  # all coding files      
    │   ├── classification      # files for classification
    │   │   ├── *.py
    │   │   └── README.md
    │   └── evaluation          # files for evaluation
    │       ├── *.py
    │       └── README.md
    ├── data 
    │   └── dataset             # mock dataset 
    │       ├── train_set.csv 
    │       └── test_set.csv
    ├── output                  # output directory 
    ├── docker-compose.yml
    ├── Dockerfile.devmodel
    ├── Dockerfile.light 
    ├── light_requirements.txt
    ├── vllm_requirements.txt
    └── README.md

Check the README files `source/classification` and `source/evaluation` for details on the coding files.

## Running 
Provided are both a `light_requirements.txt` (for code in `source/evaluation`) and a `vllm_requirements.txt` (for code in `source/classification`) file, specifiying packages and versions. During experimentation, all code in `source/classification` was run with Docker. For `source/evaluation`, Docker was only used for `error_analysis.py`. 

## Citation info 
Untill the accompanying paper is published, please reference this repo and the paper authors. 
