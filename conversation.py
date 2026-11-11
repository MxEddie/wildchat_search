"""
A class which takes a conversation from a streamed huggingface dataset and reformats it for searching, then uses regex templates to search for relevant conversations.
"""

import re
import logging

module_logger = logging.getLogger(__name__)


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

        #for i,item in enumerate(example["conversation"]):
        #    example["conversation"][i]["content"] = item["content"].replace("\n"," ").replace("\t"," ").replace('"','').replace("'","")
        self.clean_conv = example["conversation"]

    def _match_iteration(self, template_info, phrase, content, match_type, i, match_flags, set_flags):

        pattern = re.compile(phrase,flags=re.I)
        match = re.search(pattern,content)
        if match:
            match_flags[match_type] = (match.group(0), f"turn_{i}")  # store the matched phrase and the turn number
            for set_name in template_info["sets"]:
                if match_type in template_info[set_name]:
                    set_flags[set_name] = True  # mark the set as matched
            return True
        return False    

    def match_phrase(self,template_info: dict, turn_len_limit: int = None, conv_len_limit: int = None, distance_tolerance: int = 2) -> dict:
        '''Function which searches through the conversation for a match to any of the provided templates, returning a dictionary of the matches if found, or False if not. If turn_limit is provided, will also return False if any user turn exceeds the limit.
        
        Parameters:
            template_info (dict): A dictionary which contains a dictionary of templates to search for, where keys are template names and values are lists of phrases to match, and two lists of template names to check for matches in the conversation.
            turn_limit (int, optional): The maximum length of a single user turn in the conversation. If any user turn exceeds this limit, the function will return false.
            
        Returns:
            dict or bool: A dictionary of matches if found, or False if not.'''    
        #print(template_info)    

        match_flags = {key: False for key in template_info["templates"].keys()}

        set_flags = {key: False for key in template_info["sets"]}  ## Initialize flags for each set to track if a match has been found

        ## Drop categories from the templates that are not in the sets to avoid unnecessary searching
        for match_category in list(template_info["templates"].keys()):
            if not any(match_category in template_info[set_name] for set_name in template_info["sets"]) and not any(match_category in template_info[set_name] for set_name in template_info["optional_sets"]):
                del template_info["templates"][match_category]

        distance_counter = 0  # counter for the number of turns since the last match in set_A or set_B

        while not all(set_flags.values()):  ## Continue searching until a match is found in all sets
            for i, turn in enumerate(self.clean_conv):
                if turn["role"] == "user":
                    content = turn["content"]
                    if turn_len_limit and len(turn["content"]) > turn_len_limit:
                        #print(f"User content too long for conversation {self.id}")
                        content = content[:turn_len_limit]  # truncate the content to the turn length limit
                        #module_logger.warning(f"User content too long for conversation {self.id}, turn {i}, truncating to {turn_len_limit} characters.")
                    if conv_len_limit and i >= conv_len_limit:
                        #print(f"Conversation too long for conversation {self.id}")
                        module_logger.warning(f"Conversation too long for conversation {self.id}, truncating to {conv_len_limit} turns.")
                        break
                    for match_category, phrases in template_info["templates"].items():
                        if match_flags[match_category] == False:
                            #print(f"Phrases for {match_category}: {phrases}")
                            if type(phrases) is dict:
                                #print(f"Nested template found for {match_category}: {phrases}")
                                #print(f"Nested template: {phrases}")
                                match_flags[match_category] = self.match_phrase(template_info=phrases, turn_len_limit=turn_len_limit, conv_len_limit=conv_len_limit, distance_tolerance=0)  # recursive call for nested templates, do not allow distance tolerance for nested templates
                                if match_flags[match_category]:
                                    for set_name in template_info["sets"]:
                                        if match_category in template_info[set_name]:
                                            set_flags[set_name] = True  # mark the set as matched
                            else:
                                for phrase in phrases:
                                    if self._match_iteration(template_info, phrase, content, match_category, i, match_flags, set_flags):
                                        break  # exit the loop if a match is found
                    if list(set_flags.values()).count(True) < len(template_info["sets"]):  # if a match was found in any set, reset the distance counter
                        distance_counter += 1
                    if distance_counter > distance_tolerance:
                        match_flags = {key: False for key in template_info["templates"].keys()}
                        distance_counter = 0
                        set_flags = {key: False for key in template_info["sets"]}  ## Reset the set flags if the distance tolerance is exceeded
            break  ## Exit the while loop after one pass through the conversation turns

        if all(set_flags.values()):  ## If a match was found in all sets, return the match flags
            self.conversation = {f"turn_{i}": x['content'] for i, x in enumerate(self.clean_conv)}
            return { label: match_flags[label] for label in (template_info["templates"].keys()) }
              
        return False 