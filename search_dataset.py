""" 
Search script for finding conversations matching specific templates in the Wildchat and LMSys-Chat datasets. The script can filter conversations by country, limit the length of user turns, and save matched conversations to a CSV file. It also supports loading previously saved matched conversations from a file.
"""

import datasets 
from datasets import load_dataset
from tqdm import tqdm
import argparse
import torch
import re
import pandas as pd
import logging
from templates import AP4L_TEMPLATES, AIPV_TEMPLATES, TEST_TEMPLATES
import sys, os



print(f"Python version: {sys.version}")
print(f"Executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")


logging.basicConfig(level=logging.INFO, filename='search.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("===== Starting Search Script. =====")

# Check if GPU is available
if torch.cuda.is_available():
    logger.info("GPU is available")
else:
    logger.info("GPU is not available")

logging.getLogger("httpx").setLevel(logging.WARNING)
datasets.logging.set_verbosity_warning()

## Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--search", help="whether to search the dataset for conversations matching the templates", action='store_true', default=False)
parser.add_argument("--country", help="whether the search should be contained to a particular country",type=str, default=None)
parser.add_argument("--turn_limit", help="maximum length of a single turn in the conversation", type=int, default=8000)
parser.add_argument("--data_limit",help="maximum number of instances you want to take", default=None)
parser.add_argument("--dataset",help="which dataset to use",default="wildchat")
parser.add_argument("--load_output",action='store_true', help="whether to load conversations from a file rather than the dataset",default=False)
parser.add_argument("--output_file",help="path to output file for matched conversations",default=None)

## Define classes 

class FormattedConversation:
    ''' Class which takes a raw conversation from the dataset and reformats it for searching, with each turn within the conversation cleaned of newlines, tabs and quotes. Also stores the conversation ID, model, language, country and hashed IP if available.
    
    Attributes:
        country (str): The country associated with the conversation (if available).
        id (str): The unique identifier for the conversation.
        ip (str): The hashed IP address associated with the conversation (if available).
        language (str): The language of the conversation.
        model (str): The model used in the conversation.
        clean_conv (list): A list of dictionaries representing the cleaned conversation turns.
        
    Methods:
        match_phrase(templates, turn_limit=None): Searches through the conversation for matches to provided templates, returning a dictionary of matches if found, or False if not. If turn_limit is provided, it will also return False if any user turn exceeds the limit.
    '''
    def __init__(self,example,dataset):

        if dataset == "wildchat":
            self.country = example["country"]
            self.id = example["conversation_hash"]
            self.ip = example["hashed_ip"]

        elif dataset == "lmsys-chat":
            self.country = None
            self.id = example["conversation_id"]
            self.ip = None
        
        self.language = example["language"]
        self.model = example["model"]

        for i,item in enumerate(example["conversation"]):
            example["conversation"][i]["content"] = item["content"].replace("\n"," ").replace("\t"," ").replace('"','').replace("'","")
        self.clean_conv = example["conversation"]       

    def match_phrase(self,template_info: dict, turn_limit: str = None) -> dict:
        '''Function which searches through the conversation for a match to any of the provided templates, returning a dictionary of the matches if found, or False if not. If turn_limit is provided, will also return False if any user turn exceeds the limit.
        
        Parameters:
            templates (dict): A dictionary which contains a dictionary of templates to search for, where keys are template names and values are lists of phrases to match, and two lists of template names to check for matches in the conversation.
            turn_limit (int, optional): The maximum length of a single user turn in the conversation. If any user turn exceeds this limit, the function will return false.
            
        Returns:
            dict or bool: A dictionary of matches if found, or False if not.'''        
        match_flags = {key: False for key in template_info["templates"].keys()}
        
        for i, turn in enumerate(self.clean_conv):
            if turn["role"] == "user":
                if turn_limit and len(turn["content"]) > turn_limit:
                    #print(f"User content too long for conversation {self.id}")
                    return False
                for match_type, phrases in template_info["templates"].items():
                    if match_flags[match_type] == False:
                        for phrase in phrases:
                            pattern = re.compile(phrase,flags=re.I)
                            match = re.search(pattern,turn["content"])
                            if match:
                                match_flags[match_type] = match
                                break
            if any(match_flags[label] for label in template_info["set_A"]):
                if any(match_flags[label] for label in template_info["set_B"]):
                    self.conversation = " ".join([f"Person 1: {x['content']}" if x['role'] == 'user' else f"Person 2: {x['content']}" for x in self.clean_conv])
                    return { label: match_flags[label] for label in (template_info["templates"].keys()) }
                else:
                    continue
            else:
                continue 
        return False 

# Define key functions

def search_ds(country:str, template_info:dict, data_limit:int=None, datasetname: str="wildchat",turn_limit:int=None, output_file:str=None) -> pd.DataFrame:
    '''Function which seaches through a stream of the dataset, optionally up to a set limit, and writes matches to an output file, then returns a dataframe of the matches.
    
    Parameters:
        country (str): The country to filter conversations by (if provided).
        template_info (dict): A dictionary which contains a dictionary of templates to search for, where keys are template names and values are lists of phrases to match, and two lists of template names to check for matches in the conversation.
        data_limit (int, optional): The maximum number of instances to take from the dataset. If None, all instances will be taken.
        datasetname (str): The name of the dataset to search ("wildchat" or "lmsys-chat").
        turn_limit (int, optional): The maximum length of a single user turn in the conversation. If any user turn exceeds this limit, the function will return False.
    
    Returns:
        pd.DataFrame: A dataframe containing the matched conversations and their details.
    '''
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
    iterable_dataset = iter(dataset)

    rows = []
    #records = pd.DataFrame(columns=["Conversation ID","Conversation","Model","Hashed IP"])
    #for label in (template_info["set_A"] + template_info["set_B"]):
    #    records[label] = False

    for i,example in enumerate(tqdm(iterable_dataset)):
        if example["language"] != "English":
            continue 
        if country != None:
            if example["country"] != country:
                continue
        output = FormattedConversation(example,datasetname)
        match = output.match_phrase(template_info, turn_limit=turn_limit)
        if match:
            rows.append({"Conversation ID": output.id, "Conversation": output.conversation, "Model": output.model, "Hashed IP": output.ip, **{label: match[label] for label in (template_info["templates"].keys())}})
    iterable_dataset.close()
    
    del iterable_dataset, dataset

    records = pd.DataFrame(rows)
    print(f"{records.shape[0]} conversations matched phrases and templates")
    if output_file:
        records.to_csv(output_file,index=False)
    else:
        records.to_csv(f"./matched_{datasetname}.csv",index=False)

    return records


def main():
    logger.info("Script called")
    args = parser.parse_args()

    template_info = {}
    template_info["templates"] = TEST_TEMPLATES ## replace with relevant templates
    template_info["set_A"] = ["first_list"]
    template_info["set_B"] = ["second_list"]

    #template_info["templates"] = AIPV_TEMPLATES ## replace with relevant templates
    #template_info["set_A"] = ["advice","roleplay"]
    #template_info["set_B"] = ["breakup","queer","comeout"]

    if args.country == "UK":
        country = "United Kingdom"
    elif args.country == "US" or args.country == "USA":
        country = "United States"
    else:
        country = args.country 
    logger.info(f"Specified country: {country}")

    logger.info(f"Turn limit set to: {args.turn_limit}")

    logger.info(f"Data limit set to: {args.data_limit}")
    
    keep_df = None

    output_file = args.output_file if args.output_file else f"./matched_{args.dataset}.csv"

    if args.search:
        logger.info("Searching dataset for matching conversations")
        keep_df = search_ds(country,template_info,data_limit=args.data_limit,turn_limit=args.turn_limit,datasetname=args.dataset, output_file=output_file)

    if args.load_output:
        logger.info("Loading conversations from file")
        logger.info(f"Loading conversations from file: {output_file}")
        keep_df = pd.read_csv(output_file)
        logger.info(f"{keep_df.shape[0]} conversations loaded from file")

    if keep_df is not None:
        for target in template_info["templates"].keys():
            if target in keep_df.columns:
                matches = len(keep_df[keep_df[target] != False])
                logger.info(f"{target} matches: {matches}")
                print(f"{target} matches: {matches}")

if __name__ == "__main__":
    main()
    logger.info("Script run complete")
    logging.shutdown()
    sys.stdout.flush()
    os._exit(0)  ## Force exit to avoid hanging due to lingering threads in the datasets library
