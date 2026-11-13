""" 
Script for searching for relevant conversations in the Wildchat and LMSys-Chat datasets, then using an LLM to filter them for relevancy. The initial search script uses regex to search for relevant phrases in the conversations and save matched conversations to a JSON file. These matched conversations can then be filtered for relevancy using an LLM, and the results can be saved to a JSON file. 
"""

import argparse
import torch
import logging
import sys, os
import json
import yaml
from filter import run_filter
from utils import load_env_config, sample_output, select_relevant_templates
from search import run_search
import re
from datasets import Dataset


logging.basicConfig(level=logging.INFO, filename='search.log', filemode='a', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("===== Starting Search Script. =====")

# Check if GPU is available
if torch.cuda.is_available():
    logger.info("GPU is available.")
else:
    logger.warning("GPU is not available, running on CPU.")

logging.getLogger("httpx2").setLevel(logging.WARNING) ## Stop HTTPX2 from logging all Huggingface streaming call info, as this is very verbose
logging.getLogger("httpcore").setLevel(logging.WARNING)


## Set up argument parser
parser = argparse.ArgumentParser()

def _common_args(parser):
    g = parser.add_argument_group("Common arguments", "Arguments for both searching and filtering conversations.")
    g.add_argument("--output-dir",help="path to output directory for matched and filtered conversations",default=None, type=str) # Handling default elsewhere as it depends on dataset name 
    g.add_argument("--search-output-file",help="path to output file for matched conversations",default=None, type=str) # Handling default elsewhere as it depends on dataset name


def _search_args(parser):
    g = parser.add_argument_group("Search arguments", "Arguments for searching dataset(s) for relevant conversations.")
    g.add_argument("--template-set",help="which template set to use for searching",choices=["AIPV_TEMPLATES", "BROAD_TEMPLATES", "TEST_TEMPLATES"],default="AIPV_TEMPLATES")
    g.add_argument("--validate-regex",help="whether to test regex compilation for all templates before searching",action="store_true")
    g.add_argument("--dataset",help="which dataset to use",default="wildchat",choices=["wildchat","lmsys-chat"])
    g.add_argument("--country", help="whether the search should be contained to a particular country",type=str, default=None)
    g.add_argument("--data-limit",help="maximum number of instances you want to take", type=int, default=None)
    g.add_argument("--search-turn-len-limit", help="maximum length of a single turn in the conversation", type=int, default=8000)
    g.add_argument("--search-conv-turn-limit", help="maximum length of a conversation for searching in turns, if exceeded remaining turns will be truncated", type=int, default=None)
    g.add_argument("--resume-search", help="whether to resume a search from an existing output file", action="store_true")
    g.add_argument("--test-input-file", type=str, help="path to a test input file for testing the search function", default=None)


def _filter_args(parser):
    g = parser.add_argument_group("Filter arguments", "Arguments for filtering matched conversations for relevancy.")
    g.add_argument("--filter-output-file", help="path to output file for filtered conversations", default=None) # Handling default elsewhere as it depends on dataset name 
    g.add_argument("--filter-model", help="Huggingface model to use for filtering", type=str, default="Qwen/Qwen3-0.6B")
    g.add_argument("--filter-data-limit", help="maximum number of matched conversations you want to take for filtering", type=int, default=None)
    g.add_argument("--filter-turn-len-limit", help="maximum length of a single turn in the conversation for filtering. Turns truncated to reduce memory load during prompting.", type=int, default=100)
    g.add_argument("--filter-conv-turn-limit", help="maximum length of a conversation for filtering in turns, if exceeded remaining turns will be truncated", type=int, default=None)
    g.add_argument("--resume-filter", help="whether to resume filtering from an existing output file", action="store_true")


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

    ## Handling regex templates
    ## Select templates from yaml for use for searching
    with open("templates.yaml", "r", encoding="utf-8") as f:
        template_info = yaml.safe_load(f)[args.template_set]
    logging.info(f"Using template set: {args.template_set} with templates: {template_info['templates'].keys()}")

    template_info = select_relevant_templates(template_info)

    ## Test regex compilation for all templates to catch any errors early
    if args.validate_regex:
        logger.info("Testing regex compilation for all templates")
        for category, phrases in template_info["templates"].items():
            logger.info(f"Testing regex compilation for category '{category}' with phrases: {phrases}")
            for phrase in phrases:
                try:
                    pattern = re.compile(phrase, flags=re.I)
                    #logger.info(f"Phrase '{phrase}' compiled to pattern: {pattern}") # optionally log the compiled pattern for debugging
                except re.error as e:
                    logger.error(f"Regex compilation error for category '{category}' with phrase '{phrase}': {e}")
                    raise ValueError(f"Regex compilation error for category '{category}' with phrase '{phrase}': {e}")


    ## Handle country name variations
    if args.country == "UK":
        country = "United Kingdom"
    elif args.country == "US" or args.country == "USA":
        country = "United States"
    else:
        country = args.country 
    logger.info(f"Specified country: {country}")


    ## Define output file paths based on dataset name and user input
    # Make directories if they don't exist
    if args.output_dir:
        # check if the directory exists, if not create it
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            logger.info(f"Created output directory: {args.output_dir}")
    else:
        args.output_dir = "./"  # Default to current directory if no output directory is specified
    search_output_file_path = os.path.join(args.output_dir, args.search_output_file if args.search_output_file else f"matched_{args.dataset}.json")
    filter_output_file_path = os.path.join(args.output_dir, args.filter_output_file if args.filter_output_file else f"filtered_{args.dataset}.json")
    

    ## Search the dataset for matching conversations and save to a JSON file, or load from an existing JSON file
    if "search" in args.steps:
        logger.info("Searching dataset for matching conversations")
        try:
            keep_dict = run_search(output_file=search_output_file_path,template_info=template_info,datasetname=args.dataset,data_limit=args.data_limit,country=country,conv_len_limit=args.search_conv_turn_limit,turn_len_limit=args.search_turn_len_limit,resume=args.resume_search, test_input_file=args.test_input_file)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise e
    else: 
        logger.info(f"Loading conversations from file: {search_output_file_path}")
        keep_dict = json.load(open(search_output_file_path,"r"))
        logger.info(f"{len(keep_dict)} conversations loaded from file")

    ## Check if any conversations were found and raise an error if not
    assert keep_dict is not None, "Conversations not found"
    assert len(keep_dict) > 0, "No search matches found. Please check your search parameters or the output file."
    

    ## Sample the matched conversations and print a sample to the console
    sample_ids, sample_conversations = sample_output(keep_dict, sample_size=5, turn_count_max=2)
    #print(f"Sample of matched conversations: {sample_conversations}")

    ## Delete categories not in the sets to avoid unnecessary counting
    match_counts = {label: 0 for label in template_info["templates"].keys()}
    for conv_id, info in keep_dict.items():
        for label in template_info["templates"].keys():
            try:
                if info[label]:
                    match_counts[label] += 1
            except KeyError:
                pass  # If the label is not present in the info, skip it
    logger.info(f"Match counts: {match_counts}")


    ## Filter matched conversations (as a hugginface data set) using an LLM for relevancy and save to a JSON file
    if "filter" in args.steps:
        logger.info("Filtering matched conversations for relevancy using an LLM")
        try:
            run_filter(conversations=keep_dict, model_name=args.filter_model, turn_len_limit=args.filter_turn_len_limit, conv_turn_limit=args.filter_conv_turn_limit, output_file=filter_output_file_path, resume=args.resume_filter, data_limit=args.filter_data_limit)
        except ValueError as e:
            logger.error(f"Filtering failed: {e}")
            print(f"Filtering failed: {e}")


if __name__ == "__main__":
    main()
    logger.info("Script run complete")
    logging.shutdown()
    sys.stdout.flush()
    os._exit(0)  ## Force exit to avoid hanging due to lingering threads in the datasets library
