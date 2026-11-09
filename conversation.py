"""
A class which takes a conversation from a streamed huggingface dataset and reformats it for searching, then uses regex templates to search for relevant conversations.
"""

import re

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

    def match_phrase(self,template_info: dict, turn_len_limit: str = None) -> dict:
        '''Function which searches through the conversation for a match to any of the provided templates, returning a dictionary of the matches if found, or False if not. If turn_limit is provided, will also return False if any user turn exceeds the limit.
        
        Parameters:
            templates (dict): A dictionary which contains a dictionary of templates to search for, where keys are template names and values are lists of phrases to match, and two lists of template names to check for matches in the conversation.
            turn_limit (int, optional): The maximum length of a single user turn in the conversation. If any user turn exceeds this limit, the function will return false.
            
        Returns:
            dict or bool: A dictionary of matches if found, or False if not.'''        
        match_flags = {key: False for key in template_info["templates"].keys()}

        set_A_flag = False
        set_B_flag = False
        distance_tolerance = 2  # number of turns between matches in set_A and set_B to still consider a match
        distance_counter = 0  # counter for the number of turns since the last match in set_A or set_B

        while not set_A_flag and not set_B_flag:
            for i, turn in enumerate(self.clean_conv):
                if turn["role"] == "user":
                    if turn_len_limit and len(turn["content"]) > turn_len_limit:
                        #print(f"User content too long for conversation {self.id}")
                        return False
                    for match_type, phrases in template_info["templates"].items():
                        if match_flags[match_type] == False:
                            for phrase in phrases:
                                pattern = re.compile(phrase,flags=re.I)
                                match = re.search(pattern,turn["content"])
                                if match:
                                    match_flags[match_type] = (match.group(0), f"turn_{i}")  # store the matched phrase and the turn number
                                    if match_type in template_info["set_A"]:
                                        set_A_flag = True
                                    if match_type in template_info["set_B"]:
                                        set_B_flag = True
                                    break
                    if set_A_flag != set_B_flag:
                        distance_counter += 1
                    if distance_counter > distance_tolerance:
                        match_flags = {key: False for key in template_info["templates"].keys()}
                        distance_counter = 0
                        set_A_flag = False
                        set_B_flag = False
                        break
            break  ## Exit the while loop after one pass through the conversation turns

        if any(match_flags[label] for label in template_info["set_A"]) and any(match_flags[label] for label in template_info["set_B"]):
            self.conversation = {f"turn_{i}": x['content'] for i, x in enumerate(self.clean_conv)}
            return { label: match_flags[label] for label in (template_info["templates"].keys()) }
              
        return False 