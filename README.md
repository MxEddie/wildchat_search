A simple python script to search Huggingface datasets for mentions of seeking support around major social transitions.

**You must be granted access to lmsys/lmsys-chat-1m and apply for permission to access allenai/WildChat-4.8M-Full in order to search these datasets **


Uses regex to match key phrases related to seeking advice, or social roleplay.
Uses regex to match key phrases related to breakups, or coming out as queer.

#### To search dataset 

    python search_dataset --search 
    --dataset # Which dataset to search out of wildchat, lmsys 
    [--country] name_of_country # Optional, filter by select country
    [--turn_limit] limit_int # Optional, maximum length of individual turn, useful for filtering out e.g. translation, summarisation, code evaluation requests 
    [--data_limit] limit_in # Optional, number of conversations to stream up until 
    [--load_output] # Whether to load matched conversations, for analysis 
    

Filtered data related to breakups analysed in:
Eddie L. Ungless and Nishanth Sastry. 2026. I’m Thinking of Ending Things: Use of LLMs for Support During Break-ups. In Proceedings of the Extended Abstracts of the 2026 CHI Conference on Human Factors in Computing Systems, pages 1–9, New York, NY, USA. Association for Computing Machinery.




