import pandas as pd
import json
from conversation import FormattedConversation
from datasets import load_dataset
from tqdm import tqdm
import logging
import os
from typing import Optional

module_logger = logging.getLogger(__name__)

DATASETS = {
    "wildchat": "allenai/wildchat-4.8m-full",
    "lmsys-chat": "lmsys/lmsys-chat-1m"
}

CHECKPOINT_INTERVAL = 10000  # Save progress every 10,000 conversations


def _atomic_write(output_file, text):
    """Atomically overwrite the output file and record progress.
    
    Parameters:
        output_file (str): The path to the output file.
        text (str): The text content to write to the output file.
    """
    tmp = output_file + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, output_file)  # atomic overwrite

def _write_rows(output_file, rows):
    """Write the rows to the output file in JSON format.
    
    Parameters:
        output_file (str): The path to the output file.
        rows (list): The list of rows to write.
    """
    _atomic_write(output_file, json.dumps(rows, indent=4))

def save_checkpoint(output_file, rows, next_index):
    """Save the current rows and progress index to the output file and a progress file.
    
    Parameters:
        output_file (str): The path to the output file.
        rows (list): The list of rows to save.
        next_index (int): The next index to save in the progress file (e.g. where the search will resume).
    """
    _write_rows(output_file, rows)
    _atomic_write(output_file + ".progress", str(next_index))

def load_checkpoint(output_file, resume) -> tuple[dict, int]:
    """
    Load the saved rows and progress index from the output file and a progress file.
    
    Parameters:
        output_file (str): The path to the output file.
        resume (bool): Whether to resume from a previous checkpoint.
    
    Returns:
        tuple[dict, int]: A tuple containing the loaded rows and the starting index.
    """
    rows, start = {}, 0
    if resume:
        try:
            with open(output_file) as f:
                rows = json.load(f)

            progress_path = output_file + ".progress"
            with open(progress_path) as f:
                start = int(f.read().strip())
        except FileNotFoundError:
            module_logger.warning(f"Progress file or output file not found. Starting from the beginning.")
            rows, start = {}, 0
    return rows, start

def search_conversation(example: dict, datasetname: str, template_info: dict, turn_len_limit:Optional[int]=None, conv_len_limit:Optional[int]=None) -> Optional[tuple[str, dict]]:
    """Search a single conversation for matches to the provided templates. 
    
    Parameters:
        example (dict): The conversation to search.
        datasetname (str): The name of the dataset the conversation belongs to (defines expected format).
        template_info (dict): The templates to search for.
        turn_len_limit (int, optional): The maximum length of a single user turn in the conversation. If any user turn exceeds this limit, will be truncated.
        conv_len_limit (int, optional): The maximum length of a conversation for searching, if exceeded remaining turns will be skipped.

    Returns:
        tuple[str, dict]: A tuple containing the conversation ID and the match information, if a match is found, else None."""
    conv = FormattedConversation(example,datasetname)
    match = conv.match_phrase(template_info, turn_len_limit=turn_len_limit, conv_len_limit=conv_len_limit) 
    if match:
        return (conv.id, {"Conversation": conv.conversation, "Model": conv.model, "Hashed IP": conv.ip, **match})
    return None

def run_search(output_file:str,template_info:dict, datasetname:str, data_limit:Optional[int]=None,country:Optional[str]=None, turn_len_limit:Optional[int]=None, conv_len_limit:Optional[int]=None, resume:bool=False, test_input_file:str=None) -> dict:
    '''Function which seaches through a stream of the dataset, optionally up to a set limit, and writes matches to an output file, then returns a dataframe of the matches.
    
    Parameters:
        output_file (str): The path to the output file where matched conversations will be saved.
        template_info (dict): A dictionary which contains a dictionary of templates to search for, where keys are template names and values are lists of phrases to match, and two lists of template names to check for matches in the conversation.
        datasetname (str): The name of the dataset to search ("wildchat" or "lmsys-chat").
        data_limit (int, optional): The maximum number of instances to take from the dataset. If None, all instances will be taken.
        country (str): The country to filter conversations by (if provided).
        turn_len_limit (int, optional): The maximum length of a single user turn in the conversation. If any user turn exceeds this limit, the function will return False.
        conv_len_limit (int, optional): The maximum length of a conversation for searching, if exceeded remaining turns will be skipped.
        resume (bool): Whether to resume from a previous search. If True, the function will load the output file and progress file to continue searching from where it left off.
        test_input_file (str, optional): The path to a test input file containing a single conversation to test the search function. If provided, the function will only process this single conversation instead of streaming the dataset.
    
    Returns:
        dict: A dictionary containing the matched conversations and their details.
    '''

    if datasetname not in DATASETS:
        raise ValueError(f"Dataset {datasetname} not supported. Please choose either 'wildchat' or 'lmsys-chat'. This determines expected format, even if not streaming the dataset.")

    if not output_file:
        raise ValueError("Output file path must be provided.")

    module_logger.info("Output file path: %s | Dataset: %s", output_file, datasetname)

    search_kwargs = dict(datasetname=datasetname,
        template_info=template_info,
        turn_len_limit=turn_len_limit,
        conv_len_limit=conv_len_limit)

    ## Test Mode 
    if test_input_file:
        module_logger.info(f"Test input file provided: {test_input_file}. Running search on this single conversation.")
        with open(test_input_file) as f:
            example = json.load(f)
        hit = search_conversation(example, **search_kwargs)
        rows = {hit[0]: hit[1]} if hit else {}
        if output_file:
            _write_rows(output_file, rows) # Does not save progress file in test mode
        return rows


    ## Streaming mode 
    module_logger.info(f"Turn length limit: {turn_len_limit} | Conversation length limit: {conv_len_limit}")
    module_logger.info(f"Data limit set to: {data_limit}")
    module_logger.info(f"Resume search from: {resume}")
    module_logger.info(f"Filtering by country: {country}")


    if datasetname == "wildchat":
            name = "allenai/wildchat-4.8m-full"
    elif datasetname == "lmsys-chat":
        name = "lmsys/lmsys-chat-1m"
    else:
        raise ValueError(f"Dataset {datasetname} not supported. Please choose either 'wildchat' or 'lmsys-chat'.")

    rows, start_index = load_checkpoint(output_file, resume)
    
    dataset = load_dataset(name,"default",split="train",streaming=True)

    if data_limit is not None:
        dataset = dataset.take(data_limit)
    if start_index > 0:
        module_logger.info(f"Resuming search from index {start_index}")
        dataset = dataset.skip(start_index)

    next_index = start_index

    try:            
        for i,example in enumerate(tqdm(dataset, initial=next_index, total=data_limit, desc="Searching dataset")):
            try:
                if example.get("language") == "English" and (country is None or example.get("country") == country):
                    hit = search_conversation(rows, example, **search_kwargs)
                    if hit:
                        rows[hit[0]] = hit[1]
            except Exception as e:
                module_logger.error(f"Error processing conversation at index {i}: {e}")
                continue  # Skip this conversation and continue with the next one
            next_index += 1
            
            if next_index % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(output_file, rows, next_index)
                module_logger.info(f"Checkpoint saved at index {next_index}. Total matches so far: {len(rows)}")
    finally:
        save_checkpoint(output_file, rows, next_index)
        module_logger.info(f"Final checkpoint saved at index {next_index}. Total matches: {len(rows)}")
    
    return rows