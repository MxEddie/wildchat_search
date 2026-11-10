A simple python script to search Huggingface datasets for mentions of seeking support around major social transitions.

**You must be granted access to lmsys/lmsys-chat-1m and apply for permission to access allenai/WildChat-4.8M-Full in order to search these datasets**


Uses regex to match key phrases related to different categories (stored in templates.yaml)

# WIP

## To do

    - [x] Don't run evaluate on empty file
    - [x] Decide how best to handle long conversation turns (skip, truncate?)
    - [x] For love of god, add ability to save and resume search 
    - [ ] Add ability to resume filtering (e.g. split filter into n number of filtering tasks)
    - [ ] Add ability to handle arbitrary number of category sets where the conversation must match all n categories to be included

#### To search dataset 

    python search_dataset 
    search ## Whether to conduct search of dataset
    filter ## Whether to filter matched conversations using LLM 
    [--dataset] # Which dataset to search out of wildchat, lmsys 
    [--template-set] TEMPLATE_SET_NAME # Which template set to select from templates.yaml
    [--country] name_of_country # filter by select country
    [--search-turn-len-limit] int # maximum string length of individual turn, turn truncated if excess
    [--search-conv-turn-limit] int # Maximum number of turns in conversation to search before moving on
    [--data-limit] int # Number of conversations to stream up until 




Filtered data related to breakups analysed in:
Eddie L. Ungless and Nishanth Sastry. 2026. I’m Thinking of Ending Things: Use of LLMs for Support During Break-ups. In Proceedings of the Extended Abstracts of the 2026 CHI Conference on Human Factors in Computing Systems, pages 1–9, New York, NY, USA. Association for Computing Machinery.




