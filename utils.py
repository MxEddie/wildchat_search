import re
import json
from dotenv import dotenv_values, load_dotenv
import logging

module_logger = logging.getLogger(__name__)

def select_relevant_templates(template_info: dict) -> dict:
    ## Drop categories from the templates that are not in the sets to avoid unnecessary searching
    included_sets = template_info["sets"] + template_info.get("optional_sets", []) # Combine sets and optional_sets into a single dictionary
    for match_category in list(template_info["templates"].keys()):
        if not any(match_category in template_info[set_name] for set_name in included_sets):
            del template_info["templates"][match_category]
    return template_info

def parse_dict(text):
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise

def load_env_config():
    """
    Load environment variables from .env.public and .env.private files into environment. Also, store the loaded variables in a dictionary for easy access.

    Returns:
    dict: A dictionary containing the loaded environment variables.
    """
    load_dotenv(".env.public")  # Load shared development variables into environment
    module_logger.info(f"Loaded public environment variables from .env.public: {dotenv_values('.env.public')}")
    load_dotenv(".env.private")  # Load sensitive variables into environment
    module_logger.info("Loaded private environment variables from .env.private") ## do not private variables! 

def sample_output(conversations: dict, sample_size:int=5, label:str=None, turn_count_max:int=None) -> tuple:
    """
    Sample a specified number of conversations from the provided dictionary of conversations.

    Parameters:
        conversations (dict): A dictionary containing conversation data.
        sample_size (int): The number of conversations to sample. Default is 5.
        label (str): The label to filter the conversations by. If None, all conversations are sampled.
        turn_count_max (int): The maximum number of turns in a conversation to include in the sample. If None, no limit is applied.

    Returns:
        tuple: A tuple containing the sampled conversation IDs and the sampled conversations.
    """
    if turn_count_max is not None:
        short_conversations = {}
        while short_conversations == {}:
            short_conversations = {conv_id: info for conv_id, info in conversations.items() if len(info["Conversation"]) <= turn_count_max}
            turn_count_max += 1  ## Increment the turn count limit until at least one conversation is found
        if short_conversations:
            conversations = short_conversations

    if label == None:
        sample_ids = list(conversations.keys())[:sample_size]
        sampled_conversations = {conv_id: conversations[conv_id] for conv_id in sample_ids}
    else:
        true_examples = [conv_id for conv_id, info in conversations.items() if info["Evaluation"] != None and info["Evaluation"][label] == True]
        false_examples = [conv_id for conv_id, info in conversations.items() if info["Evaluation"] != None and info["Evaluation"][label] == False]
        sampled_true = {conv_id: conversations[conv_id] for conv_id in true_examples[:sample_size]}
        sampled_false = {conv_id: conversations[conv_id] for conv_id in false_examples[:sample_size]}
        sampled_conversations = {**sampled_true, **sampled_false}
        sample_ids = {"true_examples": [key for key in sampled_true.keys()], "false_examples": [key for key in sampled_false.keys()]}
    return sample_ids, sampled_conversations