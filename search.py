import json
from conversation import FormattedConversation
from datasets import load_dataset
from tqdm import tqdm
import logging
import os
from typing import Optional

module_logger = logging.getLogger(__name__)

DATASETS = {
    "wildchat": "allenai/wildchat-4.8m-full",
    "lmsys-chat": "lmsys/lmsys-chat-1m"
}

BATCH_SIZE = 1000  # Conversations per batch; progress is saved after every batch


def _atomic_write(path, text):
    """Atomically overwrite a file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def save_progress(output_file, next_index, offset):
    """Atomically record the next dataset index and the matching output file size in bytes."""
    _atomic_write(output_file + ".progress", f"{next_index} {offset}")

def load_progress(output_file, resume) -> tuple[int, int]:
    """Return (next_index, output_offset) to resume from, or (0, 0) if not resuming."""
    if resume:
        try:
            with open(output_file + ".progress") as f:
                parts = f.read().split()
            index = int(parts[0])
            # Older progress files only held the index; fall back to the current file size
            offset = int(parts[1]) if len(parts) > 1 else os.path.getsize(output_file)
            return index, offset
        except FileNotFoundError:
            module_logger.warning("Progress or output file not found. Starting from the beginning.")
    return 0, 0

def search_conversation(example: dict, datasetname: str, template_info: dict, turn_len_limit:Optional[int]=None, conv_len_limit:Optional[int]=None) -> Optional[tuple[str, dict]]:
    """Search a single conversation for matches to the provided templates. 
    
    Returns:
        tuple[str, dict]: A tuple containing the conversation ID and the match information, if a match is found, else None."""
    conv = FormattedConversation(example,datasetname)
    match = conv.match_phrase(template_info, turn_len_limit=turn_len_limit, conv_len_limit=conv_len_limit) 
    if match:
        return (conv.id, {"Conversation": conv.conversation, "Model": conv.model, "Hashed IP": conv.ip, **match})
    return None

def search_batch(batch: dict, country: Optional[str], **search_kwargs) -> dict:
    """Map function: regex-search one batch (dict of column -> list) of conversations.

    Returns a single output row for the batch: the number of conversations
    processed and the matches as newline-separated JSON lines.
    """
    keys = list(batch.keys())
    n = len(batch[keys[0]])
    lines = []
    for values in zip(*batch.values()):
        example = dict(zip(keys, values))
        try:
            if example.get("language") == "English" and (country is None or example.get("country") == country):
                hit = search_conversation(example, **search_kwargs)
                if hit:
                    lines.append(json.dumps({"id": hit[0], **hit[1]}, ensure_ascii=False))
        except Exception as e:
            module_logger.error(f"Error processing conversation: {e}")
    return {"n": [n], "hits": ["\n".join(lines)]}

def run_search(output_file:str,template_info:dict, datasetname:str, data_limit:Optional[int]=None,country:Optional[str]=None, turn_len_limit:Optional[int]=None, conv_len_limit:Optional[int]=None, resume:bool=False, test_input_file:str=None) -> int:
    '''Streams the dataset in batches, regex-searches each batch with a map function, and appends matches to a JSONL file.

    Parameters are as before. 

    Returns:
        int: The number of matches written during this run.
    '''

    if datasetname not in DATASETS:
        raise ValueError(f"Dataset {datasetname} not supported. Please choose either 'wildchat' or 'lmsys-chat'. This determines expected format, even if not streaming the dataset.")

    if not output_file:
        raise ValueError("Output file path must be provided.")

    module_logger.info("Output file path: %s | Dataset: %s", output_file, datasetname)

    search_kwargs = dict(datasetname=datasetname,
        template_info=template_info,
        turn_len_limit=turn_len_limit,
        conv_len_limit=conv_len_limit)

    ## Test Mode 
    if test_input_file:
        module_logger.info(f"Test input file provided: {test_input_file}. Running search on this single conversation.")
        with open(test_input_file) as f:
            example = json.load(f)
        hit = search_conversation(example, **search_kwargs)
        with open(output_file, "w") as f:  # Does not save progress file in test mode
            if hit:
                f.write(json.dumps({"id": hit[0], **hit[1]}, ensure_ascii=False) + "\n")
        return 1 if hit else 0

    ## Streaming mode 
    module_logger.info(f"Turn length limit: {turn_len_limit} | Conversation length limit: {conv_len_limit}")
    module_logger.info(f"Data limit set to: {data_limit} | Batch size: {BATCH_SIZE}")
    module_logger.info(f"Resume search from: {resume}")
    module_logger.info(f"Filtering by country: {country}")

    start_index, start_offset = load_progress(output_file, resume)

    if start_index > 0:
        # Discard anything written after the last checkpoint (partial or unrecorded batches)
        os.truncate(output_file, start_offset)

    dataset = load_dataset(DATASETS[datasetname], "default", split="train", streaming=True)

    if data_limit is not None:
        dataset = dataset.take(data_limit)
    if start_index > 0:
        module_logger.info(f"Resuming search from index {start_index}")
        dataset = dataset.skip(start_index)

    # Remove the original columns so each batch collapses to one small result row
    columns = dataset.column_names or list(next(iter(dataset)).keys())
    dataset = dataset.map(
        search_batch,
        batched=True,
        batch_size=BATCH_SIZE,
        remove_columns=columns,
        fn_kwargs={"country": country, **search_kwargs},
    )

    next_index, n_matches = start_index, 0
    mode = "a" if start_index > 0 else "w"  # Append when resuming, otherwise start a fresh file

    next_index, n_matches = start_index, 0
    mode = "a" if start_index > 0 else "w"  # Append when resuming, otherwise start a fresh file

    with open(output_file, mode) as out, tqdm(initial=start_index, total=data_limit, desc="Searching dataset") as pbar:
        try:
            for result in dataset:
                if result["hits"]:
                    out.write(result["hits"] + "\n")
                    n_matches += result["hits"].count("\n") + 1
                out.flush()
                os.fsync(out.fileno())
                next_index += result["n"]
                save_progress(output_file, next_index, os.fstat(out.fileno()).st_size)
                pbar.update(result["n"])
        finally:
            module_logger.info(f"Stopped at index {next_index}. Matches written this run: {n_matches}")

    return n_matches