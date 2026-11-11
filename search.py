import pandas as pd
import json
from conversation import FormattedConversation
from datasets import load_dataset
from tqdm import tqdm
import logging
import os

module_logger = logging.getLogger(__name__)


def save_checkpoint(rows, output_file, next_index):
    """Atomically overwrite the output file and record progress."""
    tmp = output_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rows, f, indent=4)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, output_file)  # atomic overwrite

    with open(output_file + ".progress", "w") as f:
        f.write(str(next_index))


def load_checkpoint(output_file, resume):
    """Return (rows, start_index). Falls back to the `resume` arg if no progress file."""
    rows, start = {}, 0
    if resume:
        if os.path.exists(output_file):
            with open(output_file) as f:
                rows = json.load(f)
        progress_path = output_file + ".progress"
        if os.path.exists(progress_path):
            with open(progress_path) as f:
                start = int(f.read().strip())
        else:
            start = resume if isinstance(resume, int) else 0
    return rows, start

def search_ds(country:str, template_info:dict, data_limit:int=None, datasetname: str="wildchat",turn_len_limit:int=None, conv_len_limit:int=None, resume:bool=False, output_file:str=None, test_input:str=None) -> dict:
    '''Function which seaches through a stream of the dataset, optionally up to a set limit, and writes matches to an output file, then returns a dataframe of the matches.
    
    Parameters:
        country (str): The country to filter conversations by (if provided).
        template_info (dict): A dictionary which contains a dictionary of templates to search for, where keys are template names and values are lists of phrases to match, and two lists of template names to check for matches in the conversation.
        data_limit (int, optional): The maximum number of instances to take from the dataset. If None, all instances will be taken.
        datasetname (str): The name of the dataset to search ("wildchat" or "lmsys-chat").
        turn_len_limit (int, optional): The maximum length of a single user turn in the conversation. If any user turn exceeds this limit, the function will return False.
        conv_len_limit (int, optional): The maximum length of a conversation for searching, if exceeded remaining turns will be skipped.
    
    Returns:
        dict: A dictionary containing the matched conversations and their details.
    '''
    module_logger.info(f"Turn length limit set to: {turn_len_limit}")
    module_logger.info(f"Data limit set to: {data_limit}")
    module_logger.info(f"Resume search from: {resume}")
    module_logger.info(f"Output file: {output_file}")
    module_logger.info(f"Searching dataset: {datasetname}")
    module_logger.info(f"Filtering by country: {country}")
    
    if datasetname == "wildchat":
        name = "allenai/wildchat-4.8m-full"
    elif datasetname == "lmsys-chat":
        name = "lmsys/lmsys-chat-1m"
    else:
        raise ValueError(f"Dataset {datasetname} not supported. Please choose either 'wildchat' or 'lmsys-chat'.")

    rows, start_index = load_checkpoint(output_file, resume)
    
    dataset = load_dataset(name,"default",split="train",streaming=True).take(data_limit) if data_limit else load_dataset(name,"default",split="train",streaming=True)

    rows = {}

    module_logger.info(f"Writing matched conversations to {output_file}")

    try:
        if test_input:
            with open(test_input) as test_input:
                i, example = 0, json.load(test_input)
            print("i:", i, "example:", example)
            output = FormattedConversation(example,datasetname)
            match = output.match_phrase(template_info, turn_len_limit=turn_len_limit, conv_len_limit=conv_len_limit)
            if match:
                rows.update({output.id: {"Conversation": output.clean_conv, "Model": output.model, "Hashed IP": output.ip, **{label: match[label] for label in (template_info["templates"].keys())}}})
        else:
            for i,example in enumerate(tqdm(dataset)):
                if i < start_index:
                    continue

                if example["language"] == "English" and (country is None or example.get("country") == country):
                    output = FormattedConversation(example,datasetname)
                    match = output.match_phrase(template_info, turn_len_limit=turn_len_limit, conv_len_limit=conv_len_limit) # Turns dictionary with template names as keys and False if no match, otherwise string of matched phrase. If no matches, returns False.
                    if match:
                        rows.update({output.id: {"Conversation": output.conversation, "Model": output.model, "Hashed IP": output.ip, **{label: match[label] for label in (template_info["templates"].keys())}}})
                
                if (i + 1) % 1000 == 0:
                    save_checkpoint(rows, output_file, i + 1)
                    module_logger.info(f"Checkpoint saved at index {i + 1}. Total matches so far: {len(rows)}")
    finally:
        save_checkpoint(rows, output_file, i + 1)
        module_logger.info(f"Final checkpoint saved at index {i + 1}. Total matches: {len(rows)}")
    
    
    del dataset ## Attempting to resolve hanging threads issue 

    return rows