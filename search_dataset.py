import datasets 
from datasets import load_dataset
from tqdm import tqdm
import argparse
import torch
import re
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO, filename='search.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("===== Starting Search Script. =====")

# Check if GPU is available
if torch.cuda.is_available():
    logger.info("GPU is available")
else:
    logger.info("GPU is not available")

logging.getLogger("httpx").setLevel(logging.WARNING)
datasets.logging.set_verbosity_warning()

## Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--search", help="whether to search the dataset for conversations matching the templates", action='store_true', default=False)
parser.add_argument("--country", help="whether the search should be contained to a particular country",type=str, default=None)
parser.add_argument("--turn_limit", help="maximum length of a single turn in the conversation", type=int, default=8000)
parser.add_argument("--data_limit",help="maximum number of instances you want to take", default=None)
parser.add_argument("--dataset",help="which dataset to use",default="wildchat")
parser.add_argument("--load_output",action='store_true', help="whether to load conversations from a file rather than the dataset",default=False)

## Define classes 

class FormattedConversation():
    ''' Class which takes a raw conversation from the dataset and reformats it for searching, with each turn within the conversation cleaned of newlines, tabs and quotes. Also stores the conversation ID, model, language, country and hashed IP if available'''
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

    def match_phrase(self,templates,turn_limit=None):
        '''Function which searches through the conversation for a match to any of the provided templates, returning a dictionary of the matches if found, or False if not. If turn_limit is provided, will also return False if any user turn exceeds the limit.'''        
        match_flags = {key: False for key in templates.keys()}
        
        for i, turn in enumerate(self.clean_conv):
            if turn["role"] == "user":
                if turn_limit and len(turn["content"]) > turn_limit:
                    #print(f"User content too long for conversation {self.id}")
                    return False
                for match_type, phrases in templates.items():
                    if match_flags[match_type] == False:
                        for phrase in phrases:
                            pattern = re.compile(phrase,flags=re.I)
                            match = re.search(pattern,turn["content"])
                            if match:
                                match_flags[match_type] = match
                                break
            if match_flags["roleplay"] or match_flags["advice"]:
                if match_flags["breakup"] or (match_flags["queer"] and match_flags["comeout"]):
                    self.conversation = " ".join([f"Person 1: {x['content']}" if x['role'] == 'user' else f"Person 2: {x['content']}" for x in self.clean_conv])
                    return { "roleplay": match_flags["roleplay"], "advice": match_flags["advice"], "breakup": match_flags["breakup"], "comeout": match_flags["comeout"] }
                else:
                    return False
            else:
                return False
        return False 

# Define key functions

def search_ds(country,templates,data_limit=None,datasetname="wildchat",turn_limit=None):
    '''Function which seaches through a stream of the dataset, optionally up to a set limit, and writes matches to an output file, then returns a dataframe of the matches'''
    if data_limit is None:
        data_limit = False
    else:
        data_limit = int(data_limit)
    
    if datasetname == "wildchat":
        name = "allenai/wildchat-4.8m-full"
    elif datasetname == "lmsys-chat":
        name = "lmsys/lmsys-chat-1m"
    else:
        raise ValueError(f"Dataset {datasetname} not supported. Please choose either 'wildchat' or 'lmsys-chat'.")

    dataset = load_dataset(name,"default",split="train",streaming=True).take(data_limit) if data_limit else load_dataset(name,"default",split="train",streaming=True)

    records = pd.DataFrame(columns=["Conversation ID","Conversation","Model","Hashed IP","roleplay","advice","breakup","comeout"])

    for i,example in enumerate(tqdm(dataset)):
        output = FormattedConversation(example,datasetname)
        if output.language != "English":
            continue 
        if country != None:
            if output.country != country:
                continue
        match = output.match_phrase(templates, turn_limit=turn_limit)
        if match:
            records.loc[len(records)] = [output.id, output.conversation, output.model, output.ip, match["roleplay"], match["advice"], match["breakup"], match["comeout"]]

    print(f"{records.shape[0]} conversations matched phrases and templates")
    records.to_csv(f"../matched_{datasetname}.csv",index=False)

    return records


def main():
    logger.info("Script called")
    args = parser.parse_args()

    # Define keywords which will be searched for 
    templates = {
        "advice": [
            r"(help|advise) me",
            r"(what |how )?(can|do|should|could) I",
            r"what'?s the [a-z]+st way to",
            r"is it [a-z]+ to",
            r"(tell|teach|show) me how (to|I)",
            r"(I('?d)* (really )*(want|need|(would )?like)|give me) (your |some |any )?(advice|help|suggestions?|thoughts?|input|ideas?)",
            r"how to do (it|that)",
            r"is (this|that|it) an? [a-z]+ idea",
            r"is my idea [a-z]+",
            r"do you have (any |some )?(advice|help|suggestions?|thoughts?|input|ideas?)",
            r"what do you think (about|of) (it|that|my|this)"
        ], 
        "roleplay": [
            r"pretend (to be|you('| a)?re?) (my|him|her|them|someone|an?)",
            r"(role( |-)?)?play as (my|him|her|them|someone|an?)",
            r"(pretend|respond|talk|act|behave|speak) (to me )*(like|as( if)?) (you( we| a|')?re? )?(my|him|her|them|someone|an?)", 
            r"practi[cs]e [a-z]*ing (to )*(my|him|her|them|someone|an?)"
        ],
        "breakup":  [
            r"break(ing)?( |-)?up with?",r"end(ing)? (it|things) with?",r"finish(ing)? (things )?with?",r"call(ing)? (it|things) off with?",r"tell(ing)? (my |a )? ([a-z]+ )+ (that )*(it('| i)?s|we('| a)?re|things are) (finished|over|through|done|thru)",r"tell(ing)? ([a-z]+ )+ (that )*I('| a)*m (going to |planning on |gonna |finna )*leav[a-z]+",r"leav(e|ing)? (my |a ) [a-z]+\W?$",r"(I|we('?ve)?) (need to|should|have to|got to|gotta|gonna)? break( |-)?up", r"call (off )*(my )*(wedding|engagement)( off )*",r"\bend (my )*(relationship|engagement)"
        ],
        "queer": [
            r"trans[a-z]*", r"(?!het)[a-z]+sexual",r"(?!cis)[a-z]+gender", r"genderfluid", r"gay", r"(a )?lesbian", r"LGBT[A-Z]*", r"pansexual", r"queer", r"same gender loving", r"a bear", r"a butch", r"a cub", r"a dyke", r"a femme", r"a stud", r"a twink", r"latinx", r"non( |-)?binary", r"enby", r"transfem[a-z]*", r"transmasc(uline)?", r"two( |-)?spirit",r"queer",r"bi",r"pan",r"homo",r"ace"],
        "comeout": [
            r"com(e|ing)? out", r"tell(ing)? ([a-z-]+ )?I('| a)?m", r"let(ting)? ([a-z]+ )+know (that )?I('| a)?m"
        ]
    }

    ## Unused search terms
    #"IPV": [
    #        r"IPV",r"abuse",r"violen(t|ce)",r"toxic",r"control[a-z]*",r"manipulat[a-z]*",r"abuse",r"isolat[a-z]+",r"threat[a-z]*",r"intimidat[a-z]+",r"stalk[a-z]*",r"harass[a-z]*",r"jealous[a-z]*",r"blam[a-z]*",r"put[a-z]* (me|you) down",r"mak[a-z]* (me|you) feel (stupid|worthless|scared)",r"check[a-z]* (my|your) (phone|emails|texts|messages|facebook|insta(gram)|IG|snap(chat)?)",r"accus[a-z]* (me|you) of cheating",r"force[a-z]* (me|you) to"
    #    ],
    #"relationship": [
    #        r"relationship",r"[a-z]friend",r"bf",r"gf",r"partner",r"wife",r"husband",r"my SO\b",r"dating",r"marriage",r"married",r"fiance(e|é)+",r"engaged",r"spouse",r"significant other",r"my (wo)?man",r"my girl",r"hubby",r"my ex\b"
    #    ], 

    if args.country == "UK":
        country = "United Kingdom"
    elif args.country == "US" or args.country == "USA":
        country = "United States"
    else:
        country = args.country 
    logger.info(f"Specified country: {country}")

    logger.info(f"Turn limit set to: {args.turn_limit}")

    logger.info(f"Data limit set to: {args.data_limit}")
    
    keep_df = None

    if args.search:
        logger.info("Searching dataset for matching conversations")
        keep_df = search_ds(country,templates,data_limit=args.data_limit,turn_limit=args.turn_limit,datasetname=args.dataset)

    if args.load_output:
        logger.info("Loading conversations from file")
        path = f"../matched_{args.dataset}.csv"
        logger.info(f"Loading conversations from file: {path}")
        keep_df = pd.read_csv(path)
        logger.info(f"{keep_df.shape[0]} conversations loaded from file")

    if keep_df is not None:
        for target in templates.keys():
            if target in keep_df.columns:
                matches = len(keep_df[keep_df[target] != False])
                logger.info(f"{target} matches: {matches}")
                print(f"{target} matches: {matches}")

if __name__ == "__main__":
    main()
    logger.info("Script run complete")
