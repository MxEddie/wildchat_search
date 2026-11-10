import pandas as pd
import json
from conversation import FormattedConversation
from datasets import load_dataset
from tqdm import tqdm
import logging

module_logger = logging.getLogger(__name__)

def search_ds(country:str, template_info:dict, data_limit:int=None, datasetname: str="wildchat",turn_len_limit:int=None) -> dict:
    '''Function which seaches through a stream of the dataset, optionally up to a set limit, and writes matches to an output file, then returns a dataframe of the matches.
    
    Parameters:
        country (str): The country to filter conversations by (if provided).
        template_info (dict): A dictionary which contains a dictionary of templates to search for, where keys are template names and values are lists of phrases to match, and two lists of template names to check for matches in the conversation.
        data_limit (int, optional): The maximum number of instances to take from the dataset. If None, all instances will be taken.
        datasetname (str): The name of the dataset to search ("wildchat" or "lmsys-chat").
        turn_len_limit (int, optional): The maximum length of a single user turn in the conversation. If any user turn exceeds this limit, the function will return False.
    
    Returns:
        dict: A dictionary containing the matched conversations and their details.
    '''
    module_logger.info(f"Turn length limit set to: {turn_len_limit}")
    module_logger.info(f"Data limit set to: {data_limit}")
    module_logger.info(f"Searching dataset: {datasetname}")
    module_logger.info(f"Filtering by country: {country}")

    if data_limit is None:
        data_limit = False
    else:
        data_limit = int(data_limit)
    
    if datasetname == "wildchat":
        name = "allenai/wildchat-4.8m-full"
    elif datasetname == "lmsys-chat":
        name = "lmsys/lmsys-chat-1m"
    else:
        raise ValueError(f"Dataset {datasetname} not supported. Please choose either 'wildchat' or 'lmsys-chat'.")

    dataset = load_dataset(name,"default",split="train",streaming=True).take(data_limit) if data_limit else load_dataset(name,"default",split="train",streaming=True)

    rows = {}

    for i,example in enumerate(tqdm(dataset)):
        if example["language"] != "English":
            continue 
        if country != None:
            if example["country"] != country:
                continue
        output = FormattedConversation(example,datasetname)
        match = output.match_phrase(template_info, turn_len_limit=turn_len_limit) # Turns dictionary with template names as keys and False if no match, otherwise string of matched phrase. If no matches, returns False.
        if match:
            rows.update({output.id: {"Conversation": output.conversation, "Model": output.model, "Hashed IP": output.ip, **{label: match[label] for label in (template_info["templates"].keys())}}})
    
    
    del dataset ## Attempting to resolve hanging threads issue 

    return rows