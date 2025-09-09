from datasets import load_dataset, Dataset
from tqdm import tqdm
import argparse
import torch
from collections import defaultdict
import re
from timeit import default_timer as timer
import pandas as pd
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 

if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available")

parser = argparse.ArgumentParser()


#parser.add_argument("--task", help="the task determines whether you are analysing coming out (comeout) or break up (breakup) conversations",type=str)
parser.add_argument("--task", help="the task determines whether you are analysing coming out (comeout) or break up (breakup) conversations",type=str, choices=["comeout","breakup"])
parser.add_argument("-if","--input_file", help="name of the input file", type=str)
parser.add_argument("-of","--output_file", help="name of the output file", type=str)
#parser.add_argument("--usecase",help="use case determines whether you are looking for roleplay or advice",default="both")
parser.add_argument("--limit",help="maximum number of instances you want to take")
args = parser.parse_args()

#csv.field_size_limit(100000000)



model_name = "allenai/OLMoE-1B-7B-0125-Instruct" #"Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

pipe = pipeline('text-generation',model = model, tokenizer = tokenizer, device_map="auto")


def full_prompt(conversation, task="breakup"):
    if task == "breakup":
        prompt = f"<|user|>\n START OF CONVERSATION. {conversation}. END OF CONVERSATION. Does this conversation include one person asking for advice about breaking up with someone? This includes roleplaying possible scenarios. Answer in a single word, yes or no. <|assistant|>\n"
    elif task == "comeout":
        prompt = f"<|user|>\n START OF CONVERSATION. {conversation}. END OF CONVERSATION. Does this conversation include one person asking for advice about coming out as LGBTQ+? This includes roleplaying possible scenarios. Answer in a single word, yes or no. <|assistant|>\n"
    return prompt

def group_has_long_turn(group):
    return any(len(str(content)) > 5000 for content in group["Content"])

def filter_conversations(df, pipe, limit=None):
    """
    Takes a dataframe, groups by the conversation ID, joins the turns with 'TURN:', applies the full prompt function,
    and classifies each conversation using the provided pipeline. Returns a list of conversation IDs to keep
    (those classified as 'Yes').
    """
    keep_ids = []
    grouped = df.groupby("Conversation ID", as_index=False).agg({"Content": 'TURN:'.join})
    grouped["Content"] = grouped["Content"].apply(full_prompt,task=args.task)
    dataset = Dataset.from_pandas(grouped)
    for idx, out in enumerate(tqdm(pipe(KeyDataset(dataset, "Content"), max_new_tokens=1))):
        decision = out[0]['generated_text'][-3:] if 'generated_text' in out[0] else str(out)[-3:]
        if "Yes" in decision:
            keep_ids.append(grouped.loc[idx, "Conversation ID"])
        if limit and idx + 1 >= int(limit):
            break
    return keep_ids  

def __main__(args):
    if args.input_file is None:
        args.input_file = f"{args.task}_both.txt"
    if args.output_file is None:
        args.output_file = f"{args.task}_filtered.txt"
    with open(args.output_file,"w") as o:
        df = pd.read_csv(args.input_file,delimiter='\t',dtype=str)
        df["Content"] = df["Content"].apply(lambda x: str(x))
        '''# Get conversation IDs to exclude
        long_turn_ids = set(
            df.groupby("Conversation ID").filter(group_has_long_turn)["Conversation ID"].unique()
        )
        df = df[~df["Conversation ID"].isin(long_turn_ids)]'''
        keep_ids = filter_conversations(df, pipe, limit=args.limit)
        print(keep_ids )
        filtered_df = df[df['Conversation ID'].isin(keep_ids)]
        filtered_df.to_csv(o, sep='\t', index=False)
        filtered_df = df[~df['Conversation ID'].isin(keep_ids)]
        o = open(f"{args.task}_reject.txt", "w")
        filtered_df.to_csv(o, sep='\t', index=False)


if __name__ == "__main__":
    __main__(args)