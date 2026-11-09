"""
This module contains the EvaluateModel class, which is responsible for assessing whether conversations in a dataset are relevant to a particular topic. 
"""

import transformers
import logging
from time import sleep
import json
import copy
from utils import parse_dict
from tqdm import tqdm
    

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
            if self.turn_count_limit is not None:
                conversation = {f"turn_{i}": conversation[turn] for i, turn in enumerate(conversation) if i < self.turn_count_limit}  ## Create a dictionary of conversation turns with keys "turn_0", "turn_1", up to the turn count limit
            ## Crop all conversation turns to 1000 characters to avoid exceeding the model's input limit
            conversation = {turn: conversation[turn][:100] for turn in conversation}
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

        new_records = copy.deepcopy(records)  ## Create a deep copy of the records to avoid modifying the original data

        for record_id, record in tqdm(records.items(), desc=f"Generating {self.key} evaluation for records", unit="record"):
            new_record = copy.deepcopy(record)  ## Create a deep copy of the record to avoid modifying the original data
            conversations = record["Conversation"]  ## Get the list of conversations from the record
            response = self._generate_conversation(conversations, system_prompt, output_count=output_count)  ## Generate the conversation output
            ## Store json object in the new records, converting from string to dict if necessary
            try:
                response = parse_dict(response)  ## Convert the response string to a dictionary using parse_dict function
            except json.JSONDecodeError:
                module_logger.error(f"Failed to parse response for record {record_id}. Response: {response}")
                response = None

            new_records[record_id]["Evaluation"] = response  ## Store the generated output in the new records  

        return new_records  ## Return the new records with the generated output




    def gen_output(self, records: dict, system_prompt=None, output_count=1, turn_count_limit:int = None, key:str = "Sexual_Content") -> dict[list[list]]:
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

        self.turn_count_limit = turn_count_limit  ## Store the turn count limit for use in generating evaluations
        self.key = key  ## Store the key for use in generating evaluation json objects

        output_dict = self.__generate(records, system_prompt, output_count)  ## Test generation to catch any errors before generating for all prompts
        #print("Output dict:", output_dict)  ## Print the output dict to check the structure of the generated responses

        return output_dict  ## Return the output dictionary containing the generated responses and any errors encountered during generation
                

