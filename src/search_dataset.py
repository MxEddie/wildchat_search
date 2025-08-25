from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
import argparse
import torch
from collections import defaultdict
import re
from timeit import default_timer as timer

if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available")

parser = argparse.ArgumentParser()

parser.add_argument("--task", help="the task determines whether you search for coming out (comeout) or break up (breakup) prompts",type=str)
parser.add_argument("--country", help="whether the search should be contained to a particular country",type=str)
parser.add_argument("-of","--output_file", help="name of the output file", type=str)
parser.add_argument("--limit",help="maximum number of instances you want to take",type=int)
args = parser.parse_args()

key_words = {"breakup":
                [],
            "comeout":
                [],
            "test":
                [r""]
            }

with open(f"/home/eddie/wildchat/{args.task}prompts.txt","r") as f:
    for line in f.readlines():
        key_words[args.task].append(re.compile(line))

class FormattedConversation():
    def __init__(self,example):
        
        self.language = example["conversation"][0]["language"]
        self.country = example["conversation"][0]["country"]
        self.id = example["conversation_hash"]

        conv_dict = defaultdict(dict)
        for i,item in enumerate(example["conversation"]):
            conv_dict[f"turn_{i}"] = {"content":item["content"].replace("\n","\\n").replace("\t","\\t"), "role":item["role"]}
        self.clean_conv = conv_dict
        
        #self.conversation = conversation

    def match_phrase(self,keywords,country=None):
        if country:
            if self.country != country:
                return False
        for turn in self.clean_conv.values():
            if turn["role"] == "user":
                roleplay_template = [r"pretend to be (my|him|her|them|someone)",r"role( |-?)play as (my|him|her|them|someone)",r"(respond|talk|act|behave) (to me )*(like|as if) you( [a-z]+|')?re (my|him|her|them|someone)",r"(respond|talk|act|behave) (to me )*(like|as) (my|him|her|them|someone)"]
                combined = "("+")|(".join(roleplay_template) + ")"
                roleplay_match = re.match(combined,turn["content"])
                if roleplay_match:
                    combined = "("+")|(".join(keywords) + ")"
                    situation_match = re.match(combined,turn["content"])
                    if situation_match:
                        return (roleplay_match, situation_match)
        return False
            

def search_ds(output,country,keywords,limit=None):
    '''Function which seaches through a stream of the Wildchat-4.8M-Full dataset, optionally up to a set limit, and writes matches to an output file'''
    print(limit)
    with open(f"/home/eddie/wildchat/{output}","w") as f:
        f.write("Conversation ID\tRole\tContent\tRoleplay\tSituation\n")
        if limit:
            dataset = load_dataset("allenai/WildChat-4.8M-Full","default",split="train",streaming=True).take(limit)
        else:
            dataset = load_dataset("allenai/WildChat-4.8M-Full","default",split="train",streaming=True)
        
        for i,example in enumerate(tqdm(dataset)):
            output = FormattedConversation(example)
            if output.language != "English":
                continue 
            if country != "none":
                match = output.match_phrase(keywords,country)
            else:
                match = output.match_phrase(keywords)      
            if match:
                t0 = timer()
                turns = [(x["role"], x["content"]) for x in output.clean_conv.values()]
                t1 = timer()
                print(f"Time to call list comprehension is {t1-t0}")
                t0 = timer()
                for turn in turns:
                    f.write(f"{output.id}\t{turn[0]}\t{turn[1]}\t{match[0]}\t{match[1]}\n")
                t1 = timer()
                print(f"Time to write turns to file is {t1-t0}")

            if i%1000==0:
                f.flush()
    f.close()

def main(args):
    print("Script called")
    if args.country == "UK":
        args.country = "United Kingdom"
    elif args.country == "US" or args.country == "USA":
        args.country = "United States"
    search_ds(args.output_file,args.country,key_words[args.task],limit=args.limit)


if __name__ == "__main__":
    t0 = timer()
    main(args)
    t1 = timer()
    print(t1-t0)
