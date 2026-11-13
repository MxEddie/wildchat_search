"""
This module contains the EvaluateModel class, which is responsible for assessing whether conversations in a dataset are relevant to a particular topic. 
"""

import os

import transformers
import logging
from time import sleep
import json
import copy
from utils import parse_dict, sample_output
from tqdm import tqdm
from datasets import load_dataset


module_logger = logging.getLogger(__name__)


class EvaluateModel:
    """
    A class used to represent a language model for evaluating conversations based on prompts and system prompts. 

    Attributes:
        name (str): The name of the model.
        model (str): The name of the model to use for generating outputs.
        hf_access_token (str): The Hugging Face access token for the model.
        retry_attempts (int): The number of times to retry a failed API call.
        kwargs (dict): Additional keyword arguments for response generation, such as max_output_tokens, temperature, etc.

    Methods:
        gen_response(prompts, system_prompt): Generates responses for a list of prompts using the specified model and system prompt, returning a DataFrame with the results.
    """
    def __init__(self, model, hf_access_token=None, retry_attempts=3, **kwargs):
        """
        Initialize the OutputModel with the specified model, family, call type, API key, Hugging Face access token, retry attempts, and additional keyword arguments.
        """

        self.model = model
        self.hf_access_token = hf_access_token
        self.retry_attempts = int(retry_attempts)
        self.kwargs = kwargs ## Handle additional keyword arguments for response generation, such as max_output_tokens, temperature, etc.

        self.client = transformers.pipeline("text-generation", model=self.model, token=self.hf_access_token)
    
    def __checkpoint_save(self, rows, output_file, next_index):
        """Atomically overwrite the output file and record progress."""
        tmp = output_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rows, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, output_file)  # atomic overwrite

        with open(output_file + ".progress", "w") as f:
            f.write(str(next_index))

    def __checkpoint_load(self, output_file, resume):
        """Return (rows, start_index). Falls back to the `resume` arg if no progress file."""
        rows, start = {}, 0
        if resume:
            if os.path.exists(output_file):
                with open(output_file) as f:
                    rows = json.load(f)
            progress_path = output_file + ".progress"
            if os.path.exists(progress_path):
                with open(progress_path) as f:
                    start = int(f.read().strip())
            else:
                start = resume if isinstance(resume, int) else 0
        return rows, start

    def __hf_generate(self, system_prompt_dict: dict, input: list[dict]) -> str:

        kwargs = dict(self.kwargs)  ## Create a copy of the kwargs to avoid modifying the original dictionary
        if "max_output_tokens" in kwargs and kwargs["max_output_tokens"] is not None:
            kwargs["max_new_tokens"] = kwargs.pop("max_output_tokens")  ## Rename max_output_tokens to max_length for Hugging Face pipeline
        input = [system_prompt_dict] + [input]  ## Combine system prompt and input messages into a single list for Hugging Face pipeline
        input = self.client.tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True, enable_thinking=False)  ## Tokenize the input for Hugging Face pipeline
        for i in range(self.retry_attempts):
            try:
                output = self.client(input, return_full_text=False, **kwargs)  ## Generate output using the Hugging Face pipeline
                return output[0]["generated_text"]  ## Return the generated text from the output
            except Exception as e:
                module_logger.error(f"Error generating response from Hugging Face model: {e}. Attempt {i+1} of {self.retry_attempts}. Retrying...")
                sleep(2 ** i)  ## Exponential backoff for retries

        raise RuntimeError(f"Failed to generate response from Hugging Face model after {self.retry_attempts} attempts.")

    def _generate_response(self, system_prompt_dict: dict, input: list[dict]) -> str:
        """
        Generate a response for a given system prompt and input using the specified model.

        Parameters:
            system_prompt_dict (dict): A dictionary containing the system prompt with keys "role" and "content".
            input (list[dict]): A list of dictionaries representing the conversation messages, each with keys "role" and "content".

        Returns:
            str: The generated response from the model.
        """
        try:
            response = self.__hf_generate(
                    system_prompt_dict=system_prompt_dict,
                    input=input,
                )
            return response
        except Exception as e:
            raise RuntimeError(f"Failed to generate response from model {self.model} after {self.retry_attempts} attempts. Error: {e}")

    def _generate_conversation(self, conversation, system_prompt, output_count=1):

        response = None

        for count in range(output_count):
            system_prompt_dict = {"role": "system", "content": system_prompt}
            if self.conv_turn_limit is not None:
                conversation = conversation[:self.conv_turn_limit]  ## Truncate the conversation to the specified turn limit
            for i, turn in enumerate(conversation):
                if self.turn_len_limit is not None and len(turn["content"]) > self.turn_len_limit:
                    conversation[i]["content"] = turn["content"][:self.turn_len_limit]  ## Truncate the turn content to the specified length limit
            input = {"role": "user", "content": f"""Evaluate the following conversation. Respond with a single valid JSON object and nothing else. No markdown, no code fences, no explanation before or after. The JSON must have exactly these keys:
            {{
            "{self.key}": <boolean>,
            "Reasoning": <string, max 2 sentences>
            }}.
            Conversation:""" + str(conversation)} 
            try:
                response = self._generate_response(system_prompt_dict, input)
            except Exception as e:
                print(f"Error generating response: {e}")
                raise RuntimeError(f"Failed to generate response from model {self.model} after {self.retry_attempts} attempts. Error: {e}")

        return response

    def __validate_response(self, response: str) -> bool:
        """
        Validate the generated response to ensure it is a valid JSON object with the expected keys and types.

        Parameters:
            response (str): The generated response from the model.

        Returns:
            bool: True if the response is valid, False otherwise.
        """
        try:
            response_dict = parse_dict(response)  ## Convert the response string to a dictionary using parse_dict function
            if not isinstance(response_dict, dict):
                raise ValueError(f"Response is not a dictionary: {response_dict}")
            if self.key not in response_dict:
                raise ValueError(f"Response does not contain the key '{self.key}': {response_dict}")
            if response_dict[self.key] not in [True, False]:
                raise ValueError(f"Response key '{self.key}' is not a boolean: {response_dict}")
            if "Reasoning" not in response_dict:
                raise ValueError(f"Response does not contain the key 'Reasoning': {response_dict}")
            return True
        except Exception as e:
            module_logger.error(f"Invalid response: {e}. Response: {response}")
            return False


    def __generate(self, records: dict, system_prompt: str, output_count=1) -> dict[list[list]]:
        """
        Generate output for input records using the specified model and system prompt. 

        Parameters:
            records (dict): A dictionary of input records, where each record is a dictionary containing the conversation data.
            system_prompt (str): The system prompt to use for generating the response. If None, a default prompt will be used.
            input_count (int): The number of inputs to expect
            output_count (int): The number of outputs to generate for each input.
        
        Returns:
            dict[list[list]]: A dictionary...
        """

        ## Update system prompt to add limit if max_output_tokens is specified in kwargs.
        output_limit = self.kwargs.get("max_output_tokens", None)
        if output_limit is not None:
            system_prompt = f"{system_prompt} Limit your response to {output_limit} tokens." 

        rows, start_index = self.__checkpoint_load(self.output_file, self.resume)

        if rows == {}:
            new_records = {}
        else:
            new_records = copy.deepcopy(rows)  ## Create a deep copy of the rows to avoid modifying the original data

        if start_index == self.data_limit:
            module_logger.info(f"All records have already been processed. Skipping generation.")
            return new_records  ## Return the new records if all records have already been processed

        try: 
            for i, example in tqdm(enumerate(records), desc=f"Generating {self.key} evaluation for records", unit="record"):
                if i < start_index:
                    continue
                if example["language"] == "English" and (self.country is None or example["country"] == self.country):
                    record_id = example["conversation_hash"] if 'conversation_hash' in example else example["conversation_id"] if 'conversation_id' in example else None
                    new_record = copy.deepcopy(example)  ## Create a deep copy of the record to avoid modifying the original data
                    conversations = example["conversation"]  ## Get the list of conversations from the record
                    response = self._generate_conversation(conversations, system_prompt, output_count=output_count)  ## Generate the conversation output
                    ## Store json object in the new records, converting from string to dict if necessary
                    try:
                        response = parse_dict(response)  ## Convert the response string to a dictionary using parse_dict function
                        if not self.__validate_response(response):
                            module_logger.error(f"Invalid response for record {record_id}. Response: {response}")
                            continue  ## Skip to the next record if the response is invalid
                        if response[self.key] is True:
                            new_records[record_id] = new_record  ## Store the new record in the new records if the response indicates a match
                            new_records[record_id]["Evaluation"] = response  ## Store the generated output in the new records
                    except json.JSONDecodeError:
                        module_logger.error(f"Failed to parse response for record {record_id}. Response: {response}")

                    if (i + 1) % 5000 == 0 and self.output_file is not None:  ## Save progress every 100 records if output_file is specified
                        self.__checkpoint_save(new_records, self.output_file, i + 1)
        finally:
            self.__checkpoint_save(new_records, self.output_file, i + 1)
            module_logger.info(f"Final checkpoint saved at index {i + 1}. Total matches: {len(new_records)}")

        return new_records  ## Return the new records with the generated output

    def gen_output(self, records: dict, system_prompt=None, output_count=1, turn_len_limit:int = None, conv_turn_limit:int = None, key:str = "Sexual_Content", output_file:str = None, resume: bool = False, data_limit: int = None,country: str = None) -> dict[list[list]]:
        """
        Generate output for input records using the specified model and system prompt. 

        Parameters:
            records (dict): A dictionary of input records, where each record is a dictionary containing the conversation data.
            system_prompt (str): The system prompt to use for generating the response. If None, a default prompt will be used.
            input_count (int): The number of inputs to expect
            output_count (int): The number of outputs to generate for each input.
        Returns:
            dict[list[list]]: A dictionary where each key is a prompt and the value is a list of lists containing the generated response and any error encountered during generation.
        """
        if not records:
            raise ValueError("No records provided for generation.")

        self.conv_turn_limit = conv_turn_limit  ## Store the conversation turn limit for use in generating evaluations
        self.turn_len_limit = turn_len_limit  ## Store the turn length limit for use in generating evaluations
        self.key = key  ## Store the key for use in generating evaluation json objects
        self.output_file = output_file  ## Store the output file for use in generating evaluation json objects
        self.resume = resume  ## Store the resume flag for use in generating evaluation json objects
        self.data_limit = data_limit  ## Store the data limit for use in generating evaluation json objects
        self.country = country  ## Store the country for use in generating evaluation json objects

        output_dict = self.__generate(records, system_prompt, output_count)  ## Test generation to catch any errors before generating for all prompts
        #print("Output dict:", output_dict)  ## Print the output dict to check the structure of the generated responses

        return output_dict  ## Return the output dictionary containing the generated responses and any errors encountered during generation
                

def filter_conversations(conversations: dict, model_name: str = "Qwen/Qwen3-0.6B", output_count: int = 1, turn_len_limit: int = None, conv_turn_limit: int = None, system_prompt: str = None, key: str = "Sexual_Content", resume: bool = False, output_file: str = None, data_limit: int = None, country: str = None) -> dict:
    model = EvaluateModel(model=model_name, hf_access_token=os.environ.get("HF_TOKEN"), retry_attempts=3)
    
    output = model.gen_output(records=conversations, system_prompt=system_prompt, output_count=output_count, turn_len_limit=turn_len_limit, conv_turn_limit=conv_turn_limit, key=key, output_file=f"{output_file}.{key}.json", resume=resume, data_limit=data_limit, country=country)  ## Generate output for the input conversations using the specified model and system prompt

    print(f"Example conversation: {next(iter(output.items()))[1]['Evaluation']}")  ## Print the evaluation of the first conversation for debugging purposes
    
    return output  ## Return the output dictionary containing the generated evaluations for all conversations

def run_filter(dataset_name, model_name, turn_len_limit, conv_turn_limit, output_file, resume, data_limit=None, country=None):

    if dataset_name == "wildchat":
        name = "allenai/wildchat-4.8m-full"
    elif dataset_name == "lmsys-chat":
        name = "lmsys/lmsys-chat-1m"
    else:
        raise ValueError(f"Dataset {dataset_name} not supported. Please choose either 'wildchat' or 'lmsys-chat'.")

    dataset = load_dataset(name,"default",split="train",streaming=True).take(data_limit) if data_limit else load_dataset(name,"default",split="train",streaming=True)

    system_prompt = """Evaluate the following conversation as to whether it relates to someone discussing romantic relations issues. Romantic relationship issues are defined as any conversation that involves conflicts, disagreements, or problems within a romantic relationship. The user may be asking for advice, venting, looking to understand their feelings or their partner's behaviour. Flag if it is related to romantic relationship issues and provide a brief reason for your decision. Respond with a single valid JSON object and nothing else. No markdown, no code fences, no explanation before or after. The JSON must have exactly these keys:
        {{
        'IssuesRelated': <boolean>,
        "Reasoning": <string, max 2 sentences>
        }}
        Rules:
            - Use double quotes for all keys and string values.
            - Do not add extra keys.
            - If you are unsure, still return the JSON with your best answer.
            - Do not include any text outside of the JSON object.
            - Do not include \\n, \\t, or any other escape characters in the JSON output.
        
        """
    try:
        filtered_ipv = filter_conversations(conversations=dataset, model_name=model_name, turn_len_limit=turn_len_limit, conv_turn_limit=conv_turn_limit, system_prompt=system_prompt, key="IssuesRelated",resume=resume, output_file=output_file, data_limit=data_limit, country=country) ## Filter in IPV contents
        json.dump(filtered_ipv, open(output_file,"w"), indent=4)
        module_logger.info(f"Filtered conversations saved to {output_file}")
        sample_ids, sample_conversations = sample_output(filtered_ipv, sample_size=5, turn_count_max=2, label=key)
        print(f"IDs of filtered conversations: {sample_ids}")
        print(f"Sample of filtered conversations: {sample_conversations}")
    except ValueError as e:
        module_logger.error(f"Filtering failed: {e}")
        print(f"Filtering failed: {e}")
        json.dump({},open(output_file,"w"), indent=4)

