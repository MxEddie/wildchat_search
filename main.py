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
import json
from model import EvaluateModel
from dotenv import dotenv_values, load_dotenv
from conversation import FormattedConversation


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
parser.add_argument("--evaluate", help="whether to evaluate the matched conversations for relevancy using an LLM", action='store_true', default=False)
parser.add_argument("--country", help="whether the search should be contained to a particular country",type=str, default=None)
parser.add_argument("--turn-len-limit", help="maximum length of a single turn in the conversation", type=int, default=8000)
parser.add_argument("--data-limit",help="maximum number of instances you want to take", default=None)
parser.add_argument("--dataset",help="which dataset to use",default="wildchat")
parser.add_argument("--output-file",help="path to output file for matched conversations",default=None)


def load_env_config():
    """
    Load environment variables from .env.public and .env.private files into environment. Also, store the loaded variables in a dictionary for easy access.

    Returns:
    dict: A dictionary containing the loaded environment variables.
    """
    load_dotenv(".env.public")  # Load shared development variables into environment
    logger.info(f"Loaded public environment variables from .env.public: {dotenv_values('.env.public')}")
    load_dotenv(".env.private")  # Load sensitive variables into environment
    logger.info("Loaded private environment variables from .env.private") ## do not log! 



def search_ds(country:str, template_info:dict, data_limit:int=None, datasetname: str="wildchat",turn_len_limit:int=None, output_file:str=None) -> pd.DataFrame:
    '''Function which seaches through a stream of the dataset, optionally up to a set limit, and writes matches to an output file, then returns a dataframe of the matches.
    
    Parameters:
        country (str): The country to filter conversations by (if provided).
        template_info (dict): A dictionary which contains a dictionary of templates to search for, where keys are template names and values are lists of phrases to match, and two lists of template names to check for matches in the conversation.
        data_limit (int, optional): The maximum number of instances to take from the dataset. If None, all instances will be taken.
        datasetname (str): The name of the dataset to search ("wildchat" or "lmsys-chat").
        turn_len_limit (int, optional): The maximum length of a single user turn in the conversation. If any user turn exceeds this limit, the function will return False.
    
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

    rows = {}

    for i,example in enumerate(tqdm(dataset)):
        if example["language"] != "English":
            continue 
        if country != None:
            if example["country"] != country:
                continue
        output = FormattedConversation(example,datasetname)
        match = output.match_phrase(template_info, turn_len_limit=turn_len_limit)
        if match:
            rows.update({output.id: {"Conversation": output.conversation, "Model": output.model, "Hashed IP": output.ip, **{label: match[label] for label in (template_info["templates"].keys())}}})
    
    
    del dataset

    json.dump(rows, open(output_file,"w"), indent=4)

    return rows


def evaluate_conversations(conversations: dict) -> dict:
    model_name="Qwen/Qwen3-0.6B"
    output_count=1
    model = EvaluateModel(model=model_name, hf_access_token=os.environ.get("HF_TOKEN"), retry_attempts=3)

    system_prompt_template = """Evaluate the following conversation as to whether it {premise}. {definition}. Flag if it contains {label} and provide a brief reason for your decision. Respond with a single valid JSON object and nothing else. No markdown, no code fences, no explanation before or after. The JSON must have exactly these keys:
    {{
    '{key}': <boolean>,
    "Reasoning": <string, max 2 sentences>
    }}
    Rules:
        - Use double quotes for all keys and string values.
        - Do not add extra keys.
        - If you are unsure, still return the JSON with your best answer.
        - Do not include any text outside of the JSON object.
        - Do not include \\n, \\t, or any other escape characters in the JSON output.
    
    """

    ## Evaluate first few messages for sexual content
    system_prompt=system_prompt_template.format(premise="contains sexual content", definition="Sexual content is defined as any conversation that involves explicit sexual acts, sexual behavior, or sexual references.", label="sexual content", key="Sexual_Content")
    
    output = model.gen_output(records=conversations, system_prompt=system_prompt, output_count=output_count, turn_count_limit=4, key="Sexual_Content")

    print(f"Example conversation: {next(iter(output.items()))[1]['Evaluation']}")  ## Print the evaluation of the first conversation for debugging purposes

    ## Filter out conversations which are sexual content
    filtered_conversations = {conv_id: info for conv_id, info in output.items() if info["Evaluation"] != None and info["Evaluation"]['Sexual_Content'] != True}
    print(f"Filtered conversations: {len(filtered_conversations)}")

    ## Evaluate whole conversation
    system_prompt=system_prompt_template.format(premise="relates to intimate partner violence (IPV)", definition="Intimate partner violence (IPV) is defined as any behavior by a partner or ex-partner that causes physical, sexual, or psychological harm to the other partner.", label="intimate partner violence (IPV)", key="IPV")
    
    output = model.gen_output(records=filtered_conversations, system_prompt=system_prompt, output_count=output_count, key="IPV")
    return output

def sample_output(conversations: dict, sample_size:int=5) -> dict:
    """
    Sample a specified number of conversations from the provided dictionary of conversations.

    Parameters:
        conversations (dict): A dictionary containing conversation data.
        sample_size (int): The number of conversations to sample. Default is 5.

    Returns:
        dict: A dictionary containing the sampled conversations.
    """
    true_examples = [conv_id for conv_id, info in conversations.items() if info["Evaluation"] != None and info["Evaluation"]['IPV'] == True]
    false_examples = [conv_id for conv_id, info in conversations.items() if info["Evaluation"] != None and info["Evaluation"]['IPV'] == False]
    sampled_true = {conv_id: conversations[conv_id] for conv_id in true_examples[:sample_size]}
    sampled_false = {conv_id: conversations[conv_id] for conv_id in false_examples[:sample_size]}
    example_ids = {"true_examples": [key for key in sampled_true.keys()], "false_examples": [key for key in sampled_false.keys()]}
    return example_ids

def main():
    logger.info("Script called")
    args = parser.parse_args()

    load_env_config()

    template_info = AIPV_TEMPLATES

    if args.country == "UK":
        country = "United Kingdom"
    elif args.country == "US" or args.country == "USA":
        country = "United States"
    else:
        country = args.country 

    logger.info(f"Specified country: {country}")
    logger.info(f"Turn length limit set to: {args.turn_len_limit}")
    logger.info(f"Data limit set to: {args.data_limit}")
    
    output_file = args.output_file if args.output_file else f"./matched_{args.dataset}.json"

    if args.search:
        logger.info("Searching dataset for matching conversations")
        keep_dict = search_ds(country,template_info,data_limit=args.data_limit,turn_len_limit=args.turn_len_limit,datasetname=args.dataset, output_file=output_file)
    else: 
        logger.info(f"Loading conversations from file: {output_file}")
        keep_dict = json.load(open(output_file,"r"))
        logger.info(f"{len(keep_dict)} conversations loaded from file")

    if keep_dict is not None:
        ## Count how many conversations have a match for each template
        match_counts = {label: 0 for label in template_info["templates"].keys()}
        for conv_id, info in keep_dict.items():
            for label in template_info["templates"].keys():
                if info[label]:
                    match_counts[label] += 1
        logger.info(f"Match counts: {match_counts}")


    ## Export evaluated conversations to a JSON file
    if args.evaluate:
        logger.info("Evaluating matched conversations for relevancy using an LLM")
        evaluated_dict = evaluate_conversations(conversations=keep_dict)
        evaluated_output_file = f"./evaluated_{args.dataset}.json"
        json.dump(evaluated_dict, open(evaluated_output_file,"w"), indent=4)
        logger.info(f"Evaluated conversations saved to {evaluated_output_file}")
        sample = sample_output(evaluated_dict, sample_size=5)
        print(f"Sample of evaluated conversations: {sample}")


if __name__ == "__main__":
    main()
    logger.info("Script run complete")
    logging.shutdown()
    sys.stdout.flush()
    os._exit(0)  ## Force exit to avoid hanging due to lingering threads in the datasets library
