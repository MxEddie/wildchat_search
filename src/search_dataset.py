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

parser.add_argument("--task", help="the task determines whether you search for coming out (comeout) or break up (breakup) prompts",type=str)
#parser.add_argument("--country", help="whether the search should be contained to a particular country",type=str)
parser.add_argument("-of","--output_file", help="name of the output file", type=str)
parser.add_argument("--limit",help="maximum number of instances you want to take")
parser.add_argument("--usecase",help="use case determines whether you are looking for roleplay and advice, or a test",default="both")
parser.add_argument("--dataset",help="which dataset to use",default="wildchat")
parser.add_argument("--from_file",action='store_true', help="whether to load conversations from a file rather than the dataset",default=False)
parser.add_argument("--match_only",help="whether to only search for phrases and not run through the model",default=True)
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
        self.identity_flag = False
        self.template_flag = False
        self.phrase_flag = False

        #conv_dict = defaultdict(dict)
        for i,item in enumerate(example["conversation"]):
            example["conversation"][i]["content"] = item["content"].replace("\n"," ").replace("\t"," ").replace('"','').replace("'","")
        self.clean_conv = example["conversation"]       

    def match_phrase(self,templates,keywords,task):
        '''Function which searches through the conversation for a match to any of the provided phrases and templates'''        
        self.template_flag = False
        self.phrase_flag = False
        self.identity_flag = False
        
        if task == "comeout":
            phrases = keywords[task][1]
            identity = keywords[task][0]
        elif task == "breakup":
            phrases = keywords[task]
            self.identity_flag = True
            

        for i, turn in enumerate(self.clean_conv):
            if turn["role"] == "user":
                if len(turn["content"]) > 8000:
                    #print(f"User content too long for conversation {self.id}")
                    return False
                if self.template_flag == False:
                    for template in templates:
                        pattern = re.compile(template[0],flags=re.I)
                        template_match = re.search(pattern,turn["content"])
                        if template_match:
                            self.template_flag = (template_match, template[1])
                            break
                if self.phrase_flag == False:
                    for phrase in phrases:
                        pattern = re.compile(phrase,flags=re.I)
                        phrase_match = re.search(pattern,turn["content"])
                        if phrase_match:
                            self.phrase_flag = phrase_match
                            break
                if self.identity_flag == False:
                    for iden in identity:
                        pattern = re.compile(iden,flags=re.I)
                        identity_match = re.search(pattern,turn["content"])
                        if identity_match:
                            self.identity_flag = identity_match
                            break

            if self.template_flag and self.phrase_flag and self.identity_flag:
                self.conversation = " ".join([f"Person 1: {x['content']}" if x['role'] == 'user' else f"Person 2: {x['content']}" for x in self.clean_conv])
                return (self.template_flag, self.phrase_flag)

        return False

# Define key functions

def full_prompt(conversation, task="breakup"):
    prompt = [{"role": "system", "content": "You are a helpful assistant that answers questions about conversations between people. You give a one-word responses. Your answer options are (yes | no | maybe) and you only answer with one of these options. All other information will be ignored."}]
    if task == "breakup":
        prompt.append({"role": "user", "content": f"Here is a conversation between two people. START OF CONVERSATION. {conversation}. END OF CONVERSATION. Does this conversation include a discussion about breaking up with someone or ending a relationship? This includes role-playing or discussing hypothetical scenarios. Give a one-word response. Answer yes or no. If unsure, answer yes."})
    elif task == "comeout":
        prompt.append({"role": "user", "content": f"Here is a conversation between two people. START OF CONVERSATION. {conversation}. END OF CONVERSATION. Does this conversation include a discussion about coming out as gay, trans or another LGBTQ+ identity? This includes role-playing or discussing hypothetical scenarios. Give a one-word response. Answer yes or no. If unsure, answer yes."})
    return prompt


def search_ds(country,templates,keywords,task,limit,datasetname="wildchat"):
    '''Function which seaches through a stream of the dataset, optionally up to a set limit, and writes matches to an output file, then returns a dataframe of the matches'''
    if limit == "None":
        limit = False
    else:
        limit = int(limit)
    
    print(templates)
    print(keywords[task])

    if datasetname == "wildchat":
        name = "allenai/wildchat-4.8m-full"
    elif datasetname == "lmsys-chat":
        name = "lmsys/lmsys-chat-1m"

    
    if limit:
        dataset = load_dataset(name,"default",split="train",streaming=True).take(limit)
    else:
        dataset = load_dataset(name,"default",split="train",streaming=True)

    keep_df = pd.DataFrame(columns=["Conversation ID","Conversation","UseCase","Model","Hashed IP"])

    for i,example in enumerate(tqdm(dataset)):
        output = FormattedConversation(example,datasetname)
        if output.language != "English":
            continue 
        if country != "none":
            print("country error")
            if output.country != country:
                continue
        match = output.match_phrase(templates,keywords,task)      
        if match:
            keep_df.loc[len(keep_df)] = [output.id, output.conversation, match[0][1], output.model, output.ip]

    print(f"{keep_df.shape[0]} conversations matched phrases and templates")
    keep_df.to_csv(f"../matched_{task}_{datasetname}.csv",index=False)

    return keep_df    


def analyse_responses(keep_df, pipe, output):
    '''Function which takes a dataframe of conversations, runs them through the model pipeline and writes accepted conversations to a file'''
    dataset = Dataset.from_pandas(keep_df)
    with open(f"../{output}","w") as f:
        f.write("Conversation ID\tContent\tUseCase\tModel\tIP Address\n")
        rejection_list = []
        check_list = []
        content = pipe(KeyDataset(dataset, "Prompt"), max_new_tokens=3)
        for idx,out in enumerate(tqdm(content)):
            decision = out[0]['generated_text'][-1]["content"]
            if re.search(r"no",decision.lower()):
                rejection_list.append(keep_df.loc[idx]['Conversation ID'])
            elif re.search(r"yes",decision.lower()):            
                f.write(f"{keep_df.loc[idx]['Conversation ID']}\t{keep_df.loc[idx]['Conversation']}\t{keep_df.loc[idx]['UseCase']}\t{keep_df.loc[idx]['Model']}\t{keep_df.loc[idx]['Hashed IP']}\n")
            else:
                check_list.append(keep_df.loc[idx]['Conversation ID'])

    with open(f"../rejected_{output}","w") as f:
        f.write(f"{len(rejection_list)} conversations rejected\n")
        for conv_id in rejection_list:
            f.write(f"{conv_id}\n")
        f.write(f"{len(check_list)} conversations unclear\n")
        for conv_id in check_list:
            f.write(f"{conv_id}\n")    


def main(args):
    print("Script called")

    # Define keywords which will be searched for 

    key_words = {"breakup": 
                [r"break(ing)?( |-)?up with?",r"end(ing)? (it|things) with?",r"finish(ing)? (things )?with?",r"call(ing)? (it|things) off with?",r"tell(ing)? (my |a )? ([a-z]+ )+ (that )*(it('| i)?s|we('| a)?re|things are) (finished|over|through|done|thru)",r"tell(ing)? ([a-z]+ )+ (that )*I('| a)*m (going to |planning on |gonna |finna )*leav[a-z]+",r"leav(e|ing)? (my |a ) [a-z]+\W?$",r"(I|we('?ve)?) (need to|should|have to|got to|gotta|gonna)? break( |-)?up", r"call (off )*(my )*(wedding|engagement)( off )*",r"\bend (my )*(relationship|engagement)"],
            "comeout":
                [[r"trans[a-z]*", r"(?!het)[a-z]+sexual",r"(?!cis)[a-z]+gender", r"genderfluid", r"gay", r"(a )?lesbian", r"LGBT[A-Z]*", r"pansexual", r"queer", r"same gender loving", r"a bear", r"a butch", r"a cub", r"a dyke", r"a femme", r"a stud", r"a twink", r"latinx", r"non( |-)?binary", r"enby", r"transfem[a-z]*", r"transmasc(uline)?", r"two( |-)?spirit",r"queer",r"bi",r"pan",r"homo",r"ace"],[r"com(e|ing)? out", r"tell(ing)? ([a-z-]+ )?I('| a)?m", r"let(ting)? ([a-z]+ )+know (that )?I('| a)?m"]],
            "test":
                [r"[\s\S]*"]
            }
    
    #if args.country == "UK":
    #    args.country = "United Kingdom"
    #elif args.country == "US" or args.country == "USA":
    #    args.country = "United States"
    country = "none" 

    usecase = args.usecase
    og_roleplay_template = [
        "pretend (to be|you('| a)?re?) (my|him|her|them|someone|an?)",
        "(role( |-)?)?play as (my|him|her|them|someone|an?)",
        "(pretend|respond|talk|act|behave|speak) (to me )*(like|as( if)?) (you( we| a|')?re? )?(my|him|her|them|someone|an?)", 
        "practi[cs]e [a-z]*ing (to )*(my|him|her|them|someone|an?)"
        ]
    roleplay_template = [(x, "roleplay") for x in og_roleplay_template]
    og_advice_template = [
        "(help|advise) me",
        "(what |how )?(can|do|should|could) I",
        "what'?s the [a-z]+st way to",
        "is it [a-z]+ to",
        "(tell|teach|show) me how (to|I)",
        "(I('?d)* (really )*(want|need|(would )?like)|give me) (your |some |any )?(advice|help|suggestions?|thoughts?|input|ideas?)",
        "how to do (it|that)",
        "is (this|that|it) an? [a-z]+ idea",
        "is my idea [a-z]+",
        "do you have (any |some )?(advice|help|suggestions?|thoughts?|input|ideas?)",
        "what do you think (about|of) (it|that|my|this)"
        ]
    advice_template = [(x, "advice") for x in og_advice_template]

    if usecase == "both":
        templates = roleplay_template + advice_template
    elif usecase == "test":
        templates = [(r"[\s\S]*","test")]

    if args.from_file:
        print("loading conversations from file")
        keep_df = pd.read_csv(f"../matched_{args.task}_{args.dataset}.csv")
        print(f"{keep_df.shape[0]} conversations loaded from file")
    else:
        keep_df = search_ds(country,templates,key_words,task=args.task,limit=args.limit,datasetname=args.dataset)
    
    if args.match_only == "True":
        print("match only flag set, exiting after matching")
        return
 
    # Truncate overly long conversations
    for i, row in keep_df.iterrows():
        if len(row["Conversation"]) > 15000:
            print(f"Conversation {row['Conversation ID']} too long, truncating")
            keep_df.at[i, "Conversation"] = row["Conversation"][:15000]

    keep_df["Prompt"] = keep_df["Conversation"].apply(full_prompt,task=args.task) #Create full prompts for each conversation

    # Set up model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gc.collect()
    torch.cuda.empty_cache()
    torch.no_grad()
    torch.inference_mode() 

    model_name = "allenai/OLMo-2-0425-1B-Instruct" #"allenai/OLMo-2-0325-32B-Instruct" #"allenai/OLMoE-1B-7B-0125-Instruct" #"Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    #tokenizer.apply_chat_template(keep_df["Prompt"]) # Apply chat template to tokenizer
    pipe = pipeline('text-generation',model = model, tokenizer = tokenizer, dtype=torch.float16,batch_size=8, device=device)

    # Generate responses and write to file
    analyse_responses(keep_df, pipe, args.output_file)


if __name__ == "__main__":
    main(args)
    print("script run complete")
