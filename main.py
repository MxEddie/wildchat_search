""" 
Script for searching for relevant conversations in the Wildchat and LMSys-Chat datasets, then using an LLM to filter them. The initial search script can filter conversations by country, limit the length of user turns, and save matched conversations to a JSON file. These matched conversations can then be filtered for relevancy using an LLM, and the results can be saved to a JSON file. 
"""

import datasets 
from datasets import load_dataset
from tqdm import tqdm
import argparse
import torch
import pandas as pd
import logging
import sys, os
import json
from conversation import FormattedConversation
from templates import TEMPLATES
from filter import filter_conversations
from utils import load_env_config, sample_output
from search import search_ds


logging.basicConfig(level=logging.INFO, filename='search.log', filemode='a', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("===== Starting Search Script. =====")

# Check if GPU is available
if torch.cuda.is_available():
    logger.info("GPU is available.")
else:
    logger.warning("GPU is not available, running on CPU.")

logging.getLogger("httpx2").setLevel(logging.WARNING) ## Stop HTTPX2 from logging all call info which is very verbose
logging.getLogger("httpcore").setLevel(logging.WARNING)


## Set up argument parser
parser = argparse.ArgumentParser()

def _common_args(parser):
    g = parser.add_argument_group("Common arguments", "Arguments for both searching and filtering conversations.")
    g.add_argument("--output-file",help="path to output file for matched conversations from search.",default=None)


def _search_args(parser):
    g = parser.add_argument_group("Search arguments", "Arguments for searching dataset(s) for relevant conversations.")
    g.add_argument("--template-set",help="which template set to use for searching",choices=["AP4L_TEMPLATES","AIPV_TEMPLATES","TEST_TEMPLATES"],default="AIPV_TEMPLATES")
    g.add_argument("--dataset",help="which dataset to use",default="wildchat")
    g.add_argument("--data-limit",help="maximum number of instances you want to take", default=None)
    g.add_argument("--search-turn-len-limit", help="maximum length of a single turn in the conversation", type=int, default=8000)
    g.add_argument("--search-conv-len-limit", help="maximum length of a conversation for searching, if exceeded conversation will be ", type=int, default=1000)
    g.add_argument("--country", help="whether the search should be contained to a particular country",type=str, default=None)


def _filter_args(parser):
    g = parser.add_argument_group("Filter arguments", "Arguments for filtering matched conversations for relevancy.")
    g.add_argument("--filter-model", help="Huggingface model to use for filtering", type=str, default="Qwen/Qwen3-0.6B")
    g.add_argument("--filter-turn-len-limit", help="maximum length of a single turn in the conversation for filtering, turns truncated to reduce memory load.", type=int, default=100)
    g.add_argument("--filter-conv-len-limit", help="maximum length of a conversation for filtering, conversations truncated to reduce memory load.", type=int, default=1000)


def parse_args():
    """
    Initialize the argument parser, load common and step-specific arguments, and parse the command-line arguments. Step-specific arguments organised for readability -- will all be available regardless of which steps are selected.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("steps",
                        nargs="+",
                        choices=["search", "filter"],
                        metavar="STEP",
                        help="Steps to execute: search dataset, filter matched conversations.")

    _common_args(parser)
    _search_args(parser)
    _filter_args(parser)

    return parser.parse_args()


def main():
    logger.info("Script called")
    
    args = parse_args()
    logging.info(f"Parsed arguments: {args}")

    load_env_config()

    ## Select the template set to use for searching
    template_info = TEMPLATES[args.template_set]
    logging.info(f"Using template set: {args.template_set} with templates: {template_info['templates'].keys()}")

    if args.country == "UK":
        country = "United Kingdom"
    elif args.country == "US" or args.country == "USA":
        country = "United States"
    else:
        country = args.country 
    logger.info(f"Specified country: {country}")
    
    output_file = args.output_file if args.output_file else f"./matched_{args.dataset}.json"

    ## Search the dataset for matching conversations and save to a JSON file, or load from an existing JSON file
    if "search" in args.steps:
        logger.info("Searching dataset for matching conversations")
        keep_dict = search_ds(country,template_info,data_limit=args.data_limit,turn_len_limit=args.search_turn_len_limit,datasetname=args.dataset)
        logger.info(f"Writing matched conversations to {output_file}")
        json.dump(keep_dict, open(output_file,"w"), indent=4)
    else: 
        logger.info(f"Loading conversations from file: {output_file}")
        keep_dict = json.load(open(output_file,"r"))
        logger.info(f"{len(keep_dict)} conversations loaded from file")

    ## Check if any conversations were found and raise an error if not
    if len(keep_dict) == 0:
        raise ValueError("No conversations found. Please check your search parameters or the output file.")
        

    ## Sample the matched conversations and print a sample to the console
    sample_ids, sample_conversations = sample_output(keep_dict, sample_size=5, turn_count_max=2)
    print(f"Sample of matched conversations: {sample_conversations}")

    ## Count how many conversations have a match for each template
    assert keep_dict is not None, "No conversations found. Please check your search parameters or the output file."
    match_counts = {label: 0 for label in template_info["templates"].keys()}
    for conv_id, info in keep_dict.items():
        for label in template_info["templates"].keys():
            if info[label]:
                match_counts[label] += 1
    logger.info(f"Match counts: {match_counts}")

    ## Filter matched conversations using an LLM for relevancy and save to a JSON file
    if "filter" in args.steps:
        logger.info("Filtering matched conversations for relevancy using an LLM")
        try:
            filtered_dict = filter_conversations(conversations=keep_dict, model_name=args.filter_model)
            filtered_output_file = f"./filtered_{args.dataset}.json"
            json.dump(filtered_dict, open(filtered_output_file,"w"), indent=4)
            logger.info(f"Filtered conversations saved to {filtered_output_file}")
            sample_ids, sample_conversations = sample_output(filtered_dict, sample_size=5, turn_count_max=2)
            print(f"IDs of filtered conversations: {sample_ids}")
            print(f"Sample of filtered conversations: {sample_conversations}")
        except ValueError as e:
            logger.error(f"Filtering failed: {e}")
            print(f"Filtering failed: {e}")


if __name__ == "__main__":
    main()
    logger.info("Script run complete")
    logging.shutdown()
    sys.stdout.flush()
    os._exit(0)  ## Force exit to avoid hanging due to lingering threads in the datasets library
