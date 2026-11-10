import pandas as pd
import json
from conversation import FormattedConversation
from datasets import load_dataset
from tqdm import tqdm
import logging

module_logger = logging.getLogger(__name__)

def search_ds(country:str, template_info:dict, data_limit:int=None, datasetname: str="wildchat",turn_len_limit:int=None, conv_len_limit:int=None, resume:bool=False, output_file:str=None) -> dict:
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
    module_logger.info(f"Resume search: {resume}")
    module_logger.info(f"Output file: {output_file}")
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

    known_len = None 

    if resume and output_file is not None:
        try:
            with open("stream_record.csv", "r") as f:
                known_data = pd.read_csv(f)
            known_len = len(known_data)
            print(f"Resuming search from existing output file: {output_file}. Found {known_len} known conversation IDs.")
            module_logger.info(f"Resuming search from existing output file: {output_file}")
        except FileNotFoundError:
            module_logger.warning(f"Output file {output_file} not found. Starting a new search.")

    if known_len and data_limit:
        if known_len >= data_limit:
            module_logger.warning(f"Number of known conversation IDs ({known_len}) exceeds data limit ({data_limit}). Loading from existing output file instead of searching the dataset.")
            return json.load(open(output_file,"r"))


    dataset = load_dataset(name,"default",split="train",streaming=True).take(data_limit) if data_limit else load_dataset(name,"default",split="train",streaming=True)

    rows = {}

    module_logger.info(f"Writing matched conversations to {output_file}")

    with open("stream_record.csv", "w" if not resume else "a") as f:
        with open(output_file, "w" if not resume else "a") as out_f:
            f.write("Index,Conversation_ID\n") if not resume else None
            for i,example in enumerate(tqdm(dataset)):
                id = example["conversation_hash"] if datasetname == "wildchat" else example["conversation_id"]
                if known_len and i < known_len:
                    continue
                f.write(f"{i},{id}\n")
                if example["language"] != "English":
                    continue 
                if country != None:
                    if example["country"] != country:
                        continue
                output = FormattedConversation(example,datasetname)
                match = output.match_phrase(template_info, turn_len_limit=turn_len_limit, conv_len_limit=conv_len_limit) # Turns dictionary with template names as keys and False if no match, otherwise string of matched phrase. If no matches, returns False.
                if match:
                    rows.update({output.id: {"Conversation": output.conversation, "Model": output.model, "Hashed IP": output.ip, **{label: match[label] for label in (template_info["templates"].keys())}}})
                if i % 100 == 0:
                    json.dump(rows, out_f, indent=4)
    f.close()
    out_f.close()
    
    
    del dataset ## Attempting to resolve hanging threads issue 

    return rows