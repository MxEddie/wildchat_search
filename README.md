A simple python script to search Huggingface datasets for mentions of specific topics. It uses regex to match key phrases related to different (stored in templates.yaml). 


**You must be granted access to lmsys/lmsys-chat-1m and apply for permission to access allenai/WildChat-4.8M-Full in order to search these datasets**

You must export HF_TOKEN to environment. 


# WIP

## To do

    - [x] Don't run evaluate on empty file
    - [x] Decide how best to handle long conversation turns (skip, truncate?)
    - [x] For love of god, add ability to save and resume search 
    - [x] Add ability to resume filtering (e.g. split filter into n number of filtering tasks)
    - [x] Add ability to handle arbitrary number of category sets where the conversation must match all n categories to be included
    - [ ] Tidy up how the user can specify the distance between matches (currently hard coded)
    - [ ] Tidy up how labeling, filtering and saving are handled 

## Using Model


#### Defining Patterns
In the yaml file, you can specify which sets of categories must all have at least one match for the conversation to be returned (e.g. you can specificy it must contain at least match from the "relationship" patterns, and a match from any of the "problems", "benefits", "regrets" patterns). You can also specify whether a category should be flagged if there is a match, but that the fuction of selection conversations is not contingent on this match (e.g. identify if there is a match from the "computer" patterns, but only select the conversation if the "relationship" and "problem" categories both have matches.)

Template Collection Structure:

    TEMPLATE_COLLECTION_KEY:
      name: NAME_OF_TEMPLATE_COLLECTION
      templates:
        relationship: # Category 
        - boyfriend # Phrase
        - girlfriend
        problems:
        - fight
        - breakup
        regrets:
        - regret
        - guilt 
        computer:
        - laptop
        - deskptop
        fruit:
        - pineapple
        - orange
    set_A: # Set of Categories
    - relationship 
    set_B: 
    - problems 
    - regrets 
    set_C:
    - computer
    sets:
    - set_A ## At least one Phrase from Set A (relationship Category) must be found to select a conversation
    - set_B ## At least one Phrase from Set B (e.g. either the problems Category or regrets Category) must be found to select a conversation
    optional_sets:
    - set_C ## Matches for Phrases in the Set C (computer Category) will be highlighted but it is irrelevent to whether a conversation is selected 
    ## Phrases in the category "fruit" will not be looked for 


The design is recursive, so each category can be its own Template Selection Structure. 

#### To search dataset 

    python main.py 
    search ## Whether to conduct search of dataset
    [--output-file] name_of_file.json # Path to file where matched conversations are saved
    [--template-set] TEMPLATE_SET_NAME # Which template set to select from templates.yaml
    [--test-regex] Bool # Whether to test the regex compilation before searching
    [--dataset] # Which dataset to search out of wildchat, lmsys 
    [--country] name_of_country # filter by select country
    [--data-limit] int # Number of conversations to stream up until 
    [--search-turn-len-limit] int # maximum string length of individual turn, turn truncated if excess
    [--search-conv-turn-limit] int # Maximum number of turns in conversation to search before moving on
    [--resume-search] Bool # Whether to resume search from existing output file
    [--test-input-file] str # Path to json file containing test conversation


#### To filter matched conversations (using LLMs)

    python main.py
    filter ## Whether to filter matched conversations using LLM 
    [--output-file] name_of_file.json # Path to file where matched conversations are saved
    [--filter-output-file] name_of_file.json
    [--filter-model] str # Name of huggingface model to use for filtering
    [--filter-data-limit] int # Max number of matched conversations to load for filtering
    [--filter-turn-len-limit] int # maximum string length of individual turn, turn truncated if excess
    [--filter-turn-len-limit] int # Maximum number of turns in conversation to include in prompt
    [--resume-filter] Bool # Whether to resume filtering from existing output file 




Regex matched data related to breakups analysed in:
Eddie L. Ungless and Nishanth Sastry. 2026. I’m Thinking of Ending Things: Use of LLMs for Support During Break-ups. In Proceedings of the Extended Abstracts of the 2026 CHI Conference on Human Factors in Computing Systems, pages 1–9, New York, NY, USA. Association for Computing Machinery.




