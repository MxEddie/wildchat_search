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
parser.add_argument("-if","--input_file", help="name of the input file", type=str)
parser.add_argument("-of","--output_file", help="name of the output file", type=str)
#parser.add_argument("--usecase",help="use case determines whether you are looking for roleplay or advice",default="both")
parser.add_argument("--limit",help="maximum number of instances you want to take")
args = parser.parse_args()

#csv.field_size_limit(100000000)



model_name = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

pipe = pipeline('text-generation',model = model, tokenizer = tokenizer, device_map="auto")


def full_prompt(conversation):
    prompt = f"User: Here is a conversation. Each turn in the conversation is started with 'TURN:'. Conversation: {conversation}. END OF CONVERSATION. Is this an example of a conversation between two agents that includes one agent asking for advice about, or trying to roleplay or practice, potentially ending a relationship? Answer in a single word, yes or no. Assisstant:"
    return prompt 

def group_has_long_turn(group):
    return any(len(str(content)) > 5000 for content in group["Content"])

def filter_conversations(df, pipe, limit=None):
    """
    Takes a dataframe, groups by the conversation ID, joins the turns with 'TURN:', applies the full prompt function,
    and classifies each conversation using the provided pipeline. Returns a list of conversation IDs to keep
    (those classified as 'No').
    """
    keep_ids = []
    grouped = df.groupby("Conversation ID", as_index=False).agg({"Content": 'TURN:'.join})
    grouped["Content"] = grouped["Content"].apply(full_prompt)
    dataset = Dataset.from_pandas(grouped)
    for idx, out in enumerate(tqdm(pipe(KeyDataset(dataset, "Content"), max_new_tokens=1))):
        decision = out[0]['generated_text'][-3:] if 'generated_text' in out[0] else str(out)[-3:]
        if "Yes" in decision:
            keep_ids.append(grouped.loc[idx, "Conversation ID"])
        if limit and idx + 1 >= int(limit):
            break
    return keep_ids  

with open(args.output_file,"w") as o:
    df = pd.read_csv(args.input_file,delimiter='\t',dtype=str)
    df["Content"] = df["Content"].apply(lambda x: str(x))

    # Get conversation IDs to exclude
    long_turn_ids = set(
        df.groupby("Conversation ID").filter(group_has_long_turn)["Conversation ID"].unique()
    )
    df = df[~df["Conversation ID"].isin(long_turn_ids)]

    keep_ids = filter_conversations(df, pipe, limit=args.limit)
    filtered_df = df[~df['Conversation ID'].isin(keep_ids)]
    filtered_df.to_csv(o, sep='\t', index=False)

'''

with open(args.output_file,"w") as o:
    df = pd.read_csv(args.input_file,delimiter='\t',dtype=str)
    print(df.Content.dtypes)
    df["Content"] = df["Content"].apply(lambda x: str(x))
    grouped = df.groupby(["Conversation ID"], as_index = False).agg({"Content":'TURN:'.join})
    grouped["Content"] = grouped["Content"].apply(lambda x: full_prompt(x))
    print(grouped.shape[0])
    dataset = Dataset.from_pandas(grouped)
    print(dataset.shape)
    #dataset = dataset.to_iterable_dataset()
    for idx,out in enumerate(tqdm(pipe(KeyDataset(dataset,"Content"), max_new_tokens=1))):
        decision = out[-3:]
        if "No" in decision:
            grouped.drop([idx], inplace = True)
        if idx == args.limit:
            break 
    
    print(grouped.shape[0])

    for i,row in grouped.iterrows():
        decision = classify(row["Content"], pipe)
        if "Yes" in decision:
            keep += row["Conversation ID"]
        if i == args.limit:
            break
    



    df = df[~df['Conversation ID'].isin(keep)]

        convo_dict = defaultdict()
        for i,row in enumerate(tqdm(reader)):
            id = row[0]

            if i == 0:
                continue 
            if not row[0] in convo_dict:
                convo_dict[row[0]] = f"TURN: {row[2]}"
            elif row[0] in convo_dict:
                convo_dict[row[0]] += f"\nTURN: {row[2]}"
            if i == args.limit:
                break 
        
        if "Yes" in decision:
            o.write(key + value +"\n")
            


            #text = tokenizer.apply_chat_template(
                [{"role":"user","content":prompt}],
                tokenize=False,
                add_generation_prompt=True)
            print(output)
            #if output == "yes":
            #    o.write(value)

'''
        