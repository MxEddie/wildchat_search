from datasets import load_dataset, Dataset
#from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import argparse
import torch
from collections import defaultdict
import re
from timeit import default_timer as timer
import pandas as pd
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines.pt_utils import KeyDataset
import gc
import json


# Check if GPU is available
if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available")

# Set up argument parser
parser = argparse.ArgumentParser()

#parser.add_argument("--country", help="whether the search should be contained to a particular country",type=str)
parser.add_argument("-of","--output_file", help="name of the output file", type=str)
parser.add_argument("--limit",help="maximum number of instances you want to take")
parser.add_argument("--dataset",help="which dataset to use",default="wildchat")
parser.add_argument("--from_file",action='store_true', help="whether to load conversations from a file rather than the dataset",default=False)
args = parser.parse_args()

# Define classes 

class FormattedConversation():
    ''' Class which takes a raw conversation from the dataset and reformats it for searching'''
    def __init__(self,example,dataset):

        if dataset == "wildchat":
            self.language = example["conversation"][0]["language"]
            self.country = example["conversation"][0]["country"]
            self.id = example["conversation_hash"]
            self.ip = example["hashed_ip"]

        elif dataset == "lmsys-chat":
            self.language = example["language"]
            self.country = "none"
            self.id = example["conversation_id"]
            self.ip = "none"    

        self.model = example["model"]
        self.relationship_flag = False
        self.breakup_flag = False
        self.ipv_flag = False

        #conv_dict = defaultdict(dict)
        for i,item in enumerate(example["conversation"]):
            example["conversation"][i]["content"] = item["content"].replace("\n"," ").replace("\t"," ").replace('"','').replace("'","")
        self.clean_conv = example["conversation"]       

    def match_phrase(self,templates,task):
        '''Function which searches through the conversation for a match to any of the provided phrases and templates'''        
        self.relationship_flag = False
        self.breakup_flag = False
        self.ipv_flag = False
        
        for i, turn in enumerate(self.clean_conv):
            if turn["role"] == "user":
                if len(turn["content"]) > 8000:
                    #print(f"User content too long for conversation {self.id}")
                    return False
                if self.relationship_flag == False:
                    for template in templates[0]:
                        pattern = re.compile(template[0],flags=re.I)
                        relationship_match = re.search(pattern,turn["content"])
                        if relationship_match:
                            self.relationship_flag = relationship_match
                            break
                if self.breakup_flag == False:
                    for phrase in templates[1]:
                        pattern = re.compile(phrase,flags=re.I)
                        breakup_match = re.search(pattern,turn["content"])
                        if breakup_match:
                            self.breakup_flag = breakup_match
                            break
                if self.ipv_flag == False:
                    for iden in templates[2]:
                        pattern = re.compile(iden,flags=re.I)
                        ipv_match = re.search(pattern,turn["content"])
                        if ipv_match:
                            self.ipv_flag = ipv_match
                            break

            if self.relationship_flag:
                self.conversation = " ".join([f"Person 1: {x['content']}" if x['role'] == 'user' else f"Person 2: {x['content']}" for x in self.clean_conv])
                return (self.relationship_flag, self.breakup_flag, self.ipv_flag)

        return False

# Define key functions

def search_ds(country,templates,datasetname="wildchat"):
    '''Function which seaches through a stream of the dataset, optionally up to a set limit, and writes matches to an output file, then returns a dataframe of the matches'''
    if limit == "None":
        limit = False
    else:
        limit = int(limit)
    
    if datasetname == "wildchat":
        name = "allenai/wildchat-4.8m-full"
    elif datasetname == "lmsys-chat":
        name = "lmsys/lmsys-chat-1m"

    if limit:
        dataset = load_dataset(name,"default",split="train",streaming=True).take(limit)
    else:
        dataset = load_dataset(name,"default",split="train",streaming=True)

    keep_df = pd.DataFrame(columns=["Conversation ID","Conversation","Relationship","Break up","IPV","Model","Hashed IP"])

    for i,example in enumerate(tqdm(dataset)):
        output = FormattedConversation(example,datasetname)
        if output.language != "English":
            continue 
        if country != "none":
            print("country error")
            if output.country != country:
                continue
        match = output.match_phrase(templates)      
        if match:
            keep_df.loc[len(keep_df)] = [output.id, output.conversation, output.relationship_flag, output.breakup_flag, output.ipv_flag, output.model, output.ip]

    print(f"{keep_df.shape[0]} conversations matched phrases and templates")
    keep_df.to_csv(f"../matched_{task}_{datasetname}.csv",index=False)

    return keep_df    


def main(args):
    print("Script called")

    # Define keywords which will be searched for 

    templates = [[r"relationship",r"[a-z]friend",r"bf",r"gf",r"partner",r"wife",r"husband",r"my SO\b",r"dating",r"marriage",r"married",r"fiance(e|é)+",r"engaged",r"spouse",r"significant other",r"my (wo)?man",r"my girl",r"hubby",r"my ex\b"],                
    [r"break(ing)?( |-)?up with?",r"end(ing)? (it|things) with?",r"finish(ing)? (things )?with?",r"call(ing)? (it|things) off with?",r"tell(ing)? (my |a )? ([a-z]+ )+ (that )*(it('| i)?s|we('| a)?re|things are) (finished|over|through|done|thru)",r"tell(ing)? ([a-z]+ )+ (that )*I('| a)*m (going to |planning on |gonna |finna )*leav[a-z]+",r"leav(e|ing)? (my |a ) [a-z]+\W?$",r"(I|we('?ve)?) (need to|should|have to|got to|gotta|gonna)? break( |-)?up", r"call (off )*(my )*(wedding|engagement)( off )*",r"\bend (my )*(relationship|engagement)"],
    [r"IPV",r"abuse",r"violen(t|ce)",r"toxic",r"control[a-z]*",r"manipulat[a-z]*",r"abuse",r"isolat[a-z]+",r"threat[a-z]*",r"intimidat[a-z]+",r"stalk[a-z]*",r"harass[a-z]*",r"jealous[a-z]*",r"blam[a-z]*",r"put[a-z]* (me|you) down",r"mak[a-z]* (me|you) feel (stupid|worthless|scared)",r"check[a-z]* (my|your) (phone|emails|texts|messages|facebook|insta(gram)|IG|snap(chat)?)",r"accus[a-z]* (me|you) of cheating",r"force[a-z]* (me|you) to"]]

    #if args.country == "UK":
    #    args.country = "United Kingdom"
    #elif args.country == "US" or args.country == "USA":
    #    args.country = "United States"
    country = "none" 

    if args.from_file:
        print("loading conversations from file")
        keep_df = pd.read_csv(f"../matched_{args.task}_{args.dataset}.csv")
        print(f"{keep_df.shape[0]} conversations loaded from file")
    else:
        keep_df = search_ds(country,templates,limit=args.limit,datasetname=args.dataset)
    
    
    print("match only flag set, exiting after matching")
    return

if __name__ == "__main__":
    main(args)
    print("script run complete")
