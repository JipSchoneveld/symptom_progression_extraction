"""
All functions for running LLMs through vLLM
"""

# pyright: reportMissingImports=false, reportMissingModuleSource=false
import sys, json, re, os 
import pandas as pd  # pyright: ignore[]
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from decouple import config 
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from functools import partial

OUTPUT_DIR = "output" # type: ignore
global PROMPTS

class LLM_classifier():
    """
    LLM classifier for symptom progression extraction 
    """
    batch_size = 50 #notes per batch <- only used when saving intermediate results  

    # lookup dict from label to int 
    char2i = {
        'A': 0,
        'B': -1,
        'C': 1,
        'D': 100
    }
    i2char = {v: k for k,v in char2i.items()}

    def __init__(self, loaded_model, model_loc: str, prompts: dict, x_train: pd.Series, y_train: pd.Series, x_test, cat: int, output_dir: str, temp: bool, seed = 42, intermediate_save = False) -> None:
        """
        Create an LLM classifier for a specific category to run on the train set, or test set when given. 

        Parameters: 
        loaded_model (str): pre-loaded llm model
        model_loc (str): hugging face llm path
        prompts (dict[str, str]): all prompts to run, indexed with their prompt ID
        x_train (pd.Series): input text from train set with original indices 
        y_train (pd.Series): gold output labels with original indices 
        x_test (pd.Series): input text from test set with original indices, if None, predictions are run on train 
        cat (int): symptom category 
        output_dir (str): where to save intermediate results 
        temp (int): temperature 
        seed (int): seed 
        intermediate_save (Bool): whether to save intermediate results 
        """

        self.llm = loaded_model
        self.model_name = model_loc.split('/')[1]
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.cat = cat
        self.intermediate_save = intermediate_save
        self.output_dir = output_dir

        self.params = self.llm.get_default_sampling_params()
        self.params.seed = seed
        self.params.max_tokens = 8129
        self.params.temperature = temp
        if temp == 0:
            self.params.top_p = 1.0
            self.params.top_k = 0


        self.tokenizer = AutoTokenizer.from_pretrained(model_loc)

        # predictions dictionary
        self.predictions = defaultdict(dict)

        self.prompts = prompts
        
        # if test set is given, use x_train 
        if self.x_test is not None:
            self.X = self.x_test
            self.dataset = 'test' 
        else:
            self.X = self.x_train
            self.dataset = 'train'

        self.model_id = f'{self.model_name}{f"@{seed}" if seed != 42 else ""}#{self.dataset}_cat{cat}'

        if intermediate_save:
            self.pickle_path = os.path.join(OUTPUT_DIR, output_dir, f'predictions_{self.model_id}.pkl') #where to save intermediate results  
    
    @staticmethod
    def convert_answer(s):
        """
        Converts a raw output string to the desired label format -> integer corresponding to label or 13 if unconvertible answer 

        Parameters: 
        s (str): raw output string 

        Returns: 
        (int): corresponding integer
        """
        s = re.sub(r".*?</think>", "", s, flags=re.DOTALL) #remove think block 

        s = re.findall(r'\b[ABCD]\b', s) #find all standalone A, B, C, or D 
        s = list(set(s)) # the answer can contain the output token twice 
        if len(s) == 0 or len(s)>1: #if no ABCD or conflicting, output none
            return 13 # unqiue interger for none output 
        else:
            return LLM_classifier.char2i[s[0]] # return unique integer for found letter 


    def prompt_LLM(self, prompt_id, prompt):
        """
        Prompt the llm and collect the output. 

        Parameters: 
        prompt_id (str): ID of specific prompt to use 
        prompt (str): specific prompt to use 

        Returns: 
        note_id, generated text (int, str): the unique note id and the raw LLM response 
        """
        
        note_ids = self.X.keys() 
        notes = self.X.values

        all_prompts = [prompt.format(note) for note in notes] #format all prompts -> insert note 


        # format prompt for specific llm 
        messages_list = [
            [  {"role": "user", "content": prompt}  ] for prompt in all_prompts
        ]
        formatted_prompts = self.tokenizer.apply_chat_template(
            messages_list,
            tokenize=False,
            add_generation_prompt=True
        )

        with open(os.path.join(OUTPUT_DIR, self.output_dir, f'used_prompts_{self.model_id}.txt'), 'a') as f:
            redacted_prompts = [
                re.sub(r"<start clinical note>.*<end clinical note>", r"<start clinical note>\nNOTE HERE\n<end clinical note>", formatted_prompts[0], flags = re.DOTALL)
            ] #actual note is removed for privacy preservation
            f.writelines(redacted_prompts)
            f.write('')

        outputs = self.llm.generate(formatted_prompts, self.params, use_tqdm = partial(tqdm, position= 3, file = sys.stdout, leave = False, dynamic_ncols= True))

        for id, output in zip(note_ids, outputs):
            output_text = output.outputs[0].text
            self.predictions[prompt_id][id] = output_text
        
        return self.predictions
    
    def predict(self):
        """
        Run the LLM classifier to make all predictions 
        """
        # select the correct prompt

        global PROMPTS
        for prompt_id, prompt in tqdm(self.prompts.items(), desc="prompts", position=2, file=sys.stdout, leave = False, dynamic_ncols=True):
            
            PROMPTS.append(prompt_id + '\n' + prompt + '\n')
        
            self.prompt_LLM(prompt_id, prompt)
            
        return self.predictions


def make_llm_prediction(loaded_model, model_loc, prompts: dict, x_train: pd.Series, y: pd.Series, cat, output_dir, temp, seed, x_test = None):
    """
    Makes a prediction for all x_train or x_test using the specified model and prompts. By default, predictions are made on x_train, but when given x_test, predictions are only made on this set. 

    Parameters:
    loaded_model (str): pre-loaded llm model
    model_loc (str): hugging face llm path
    prompts (dict[str, str]): all prompts to run, indexed with their prompt ID
    x_train (pd.Series): train text from notes with NoteID as index
    y (pd.Series): train labels with NoteID as index
    cat (int): the category to make predictions for 
    output_dir (str): path to save raw predictions 
    temp (int): temperature
    x_test (pd.Series / None): test text from notes with NoteID as index
    
    Returns: 
    predicted_labels (dict[str, int]): all predicted labels as a dictionary of: NoteID -> predicted label 
    """

    global PROMPTS
    PROMPTS = [] #used for saving all used prompts (for documentation purposes)
    classifier = LLM_classifier(loaded_model, model_loc, prompts, x_train, y, x_test, cat, output_dir, temp, seed)

    predictions = classifier.predict()

    try:  
        predictions = pd.DataFrame(predictions)
    except:
        predictions = pd.Series(predictions)
    predictions = predictions.fillna("")
    predictions.to_csv(os.path.join(OUTPUT_DIR, output_dir, f'raw_output_{classifier.model_id}.csv')) #write raw output 

    # convert raw output to desired label and output format 
    predicted_labels = predictions.map(LLM_classifier.convert_answer)
    predicted_labels = predicted_labels.to_dict()

    model_name = classifier.model_id.split('#')[0].split('@')[0]

    with open(os.path.join(OUTPUT_DIR, output_dir, f'used_prompt_template_{model_name}.txt'), 'a') as f:
        f.write(str(cat) + '\n')
        f.writelines(PROMPTS)
        f.write('')

    with open(os.path.join(OUTPUT_DIR, output_dir, 'config.txt'), 'a') as f: 
        f.write(f"\n{classifier.params}")

    return predicted_labels