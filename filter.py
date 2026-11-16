import json
import logging
import os
from itertools import islice

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import IterableDataset, load_dataset

from utils import parse_dict

from tqdm import tqdm

module_logger = logging.getLogger(__name__)

TURN_LEN_LIMIT = 800
CONV_TURN_LIMIT = 8
BATCH_SIZE = 2


# File Handling Functions
def _append_jsonl(path: str, rows: list[dict])-> int:
    """Append rows as JSON lines, fsync, and return the file's new size in bytes.
    
    Parameters:
        path (str): The path to the JSONL file.
        rows (list): A list of dictionaries to append to the file.
    
    Returns:
        int: The new size of the file in bytes after appending."""
    if rows:
        with open(path, "ab") as f:
            for row in rows:
                f.write((json.dumps(row, ensure_ascii=False, default=str) + "\n").encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
    return os.path.getsize(path) if os.path.exists(path) else 0


def _save_progress(path: str, state: dict):
    """Atomically overwrite the small progress file.
    
    Parameters:
        path (str): The path to the progress file.
        state (dict): A dictionary containing the progress state to save.
    """
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_progress(output_file: str, resume: bool) -> dict:
    """Return the progress state from the .progress file, or a default state if not resuming. On resume, both JSONL files are truncated to the byte offsets recorded after the last finished batch, which discards anything written by a batch that crashed mid-way
    
    Parameters:
        output_file (str): The path to the output file.
        resume (bool): Whether to resume from a previous checkpoint.
    
    Returns:
        dict: A dictionary containing the progress state, including the next batch index, next start index, and byte offsets for the output and seen files.
    """

    progress_path = output_file + ".progress"

    if resume and os.path.exists(progress_path):
        with open(progress_path) as f:
            state = json.load(f)
        if os.path.exists(output_file) and os.path.getsize(output_file) > state["output_bytes"]:
            module_logger.warning(f"Truncating {output_file} from {os.path.getsize(output_file)} to {state['output_bytes']} bytes.")
            with open(output_file, "ab") as o:
                o.truncate(state["output_bytes"])
        module_logger.info(f"Resuming at batch {state['next_batch']} (example {state['next_start']}).")
        return state

    if resume:
        module_logger.warning("No progress file found. Starting from the beginning.")
    for p in (output_file, progress_path):
        if os.path.exists(p):
            os.remove(p)
    return {"next_batch": 0, "next_start": 0, "output_bytes": 0, "seen_bytes": 0}


def iter_batches(dataset, batch_size: int, start: int=0, data_limit: int=None):
    """Yield lists of examples, working for both streaming and map-style datasets.
    
    Parameters:
        dataset (IterableDataset or Dataset): The dataset to iterate over.
        batch_size (int): The number of examples per batch.
        start (int): The index to start from in the dataset.
        data_limit (int or None): The maximum number of examples to process. If None, there is no limit.
    
    Yields:
        list: A list of examples from the dataset, with length up to batch_size.
    """
    if isinstance(dataset, IterableDataset):
        if data_limit is not None:
            dataset = dataset.take(data_limit)
        it = iter(dataset.skip(start))
    else:
        end = len(dataset) if data_limit is None else min(data_limit, len(dataset))
        it = iter(dataset.select(range(start, end)))
    while True:
        chunk = list(islice(it, batch_size))
        if not chunk:
            return
        yield chunk

def conversation_to_turns(conversation) -> list[str]:
    """Normalise a dataset conversation into a list of turn strings."""
    if isinstance(conversation, dict):
        return [str(t) for t in conversation.values()]
    return [(str(t["role"]), str(t["content"])) if isinstance(t, dict) else str(t) for t in conversation]


def build_prompt(row: dict, conversation_column: str, key: str, conv_turn_limit: int, turn_len_limit: int) -> str:
    turns = conversation_to_turns(row[conversation_column])[:conv_turn_limit]
    turns = {f"turn_{i}_{text[0]}": text[1][:turn_len_limit] for i, text in enumerate(turns)}
    return (
        "Evaluate the following conversation for whether the user is discussing issues in their romantic relationship. Respond with a single valid JSON object and nothing else. "
        "No markdown, no code fences, no explanation before or after. "
        "The JSON must have exactly these keys:\n"
        "{\n"
        f'"{key}": <boolean>,\n'
        '"Reasoning": <string, max 2 sentences>\n'
        "}\n"
        f"Conversation: {json.dumps(turns, ensure_ascii=False)}"
    )


def validate_model_output(outputs: list, key: str) -> list[dict]:
    """Parse each output into a dict with given {key} and generated 'Reasoning'. Validation failures are logged and replaced with a dict containing {key: None} and the error message in 'Reasoning'.
    
    Parameters:
        outputs (list): A list of model output strings to validate.
        key (str): The expected key in the parsed output dict.
        
    Returns:
        list: A list of validated dictionaries, each containing the expected key and reasoning.
    """
    validated = []
    for output in outputs:
        try:
            parsed = parse_dict(output)
            if not isinstance(parsed, dict) or key not in parsed:
                raise ValueError(f"missing key '{key}'")
            parsed.setdefault("Reasoning", "")
            validated.append(parsed)
        except Exception as e:
            module_logger.error(f"Error parsing model output: {output!r}. Error: {e}")
            validated.append({key: None, "Reasoning": f"Error parsing model output: {e}"})
    return validated


class EvaluationRunner:
    def __init__(self, model_name: str, key: str):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        if self.device == "cuda":
                    print("Using CUDA for inference.")
                    print(f"dtype: {dtype}")
                    free, total = torch.cuda.mem_get_info()
                    print(f"Using {self.device} with {free/1e9:.2f} GB free of {total/1e9:.2f} GB total.")

        kwargs = dict(cache_dir=os.environ.get("HF_HOME"), token=os.environ.get("HF_TOKEN"))
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **kwargs).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", **kwargs)
        self.key = key

        if self.device == "cuda":
            print("Using CUDA for inference.")
            print(f"dtype: {dtype}")
            free, total = torch.cuda.mem_get_info()
            print(f"Using {self.device} with {free/1e9:.2f} GB free of {total/1e9:.2f} GB total.")

    @torch.inference_mode()
    def _evaluate(self, prompts: list[str], system_prompt: str) -> list[dict]:
        """Run one batch of user prompts through the model; return validated dicts.
        
        Parameters:
            prompts (list): A list of user prompt strings to evaluate.
            system_prompt (str): The system prompt to prepend to each user prompt.
            
        Returns:
            list: A list of validated dictionaries containing the expected key and reasoning.
        """
        chats = [
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False if 
            self.model_name.startswith("Qwen") else None)
            for p in prompts
        ]
        # No truncation to retain generation prompt.
        # Prompt length bound by turn_len_limit and conv_turn_limit
        inputs = self.tokenizer(chats, return_tensors="pt", padding=True).to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        del inputs, outputs, new_tokens
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return validate_model_output(decoded, self.key)

    def label_and_filter(self, dataset, system_prompt, turn_len_limit, conv_turn_limit, output_file,resume=False, conversation_column="conversation", data_limit=None, batch_size=16):
        """Label every conversation batch by batch.

        After each batch: positives are appended to {output_file} (JSONL), progress is saved to {output_file}.progress.

        Parameters:
            dataset (IterableDataset or Dataset): The dataset to process.
            system_prompt (str): The system prompt to prepend to each user prompt.
            turn_len_limit (int): The maximum length of each turn in characters.
            conv_turn_limit (int): The maximum number of turns in a conversation.
            output_file (str): The path to the output JSONL file.
            resume (bool): Whether to resume from a previous checkpoint.
            conversation_column (str): The name of the column containing the conversation data.
            data_limit (int or None): The maximum number of examples to process. If None, there is no limit.
            batch_size (int): The number of examples per batch.

        """
        progress_path = output_file + ".progress"

        state = load_progress(output_file, resume)
        start, batch_idx = state["next_start"], state["next_batch"]

        n_pos = 0
        for batch in tqdm(iter_batches(dataset, batch_size, start=start, data_limit=data_limit),                         desc="Processing batches", unit="batch"):
            prompts = [build_prompt(ex, conversation_column, self.key, conv_turn_limit, turn_len_limit)
                       for ex in batch]
            results = self._evaluate(prompts, system_prompt)

            positives = []
            for ex, res in zip(batch, results):
                if res[self.key] is True:
                    row = dict(ex)
                    row[self.key] = True
                    row["Reasoning"] = res["Reasoning"]
                    positives.append(row)

            state["output_bytes"] = _append_jsonl(output_file, positives)

            start += len(batch)
            batch_idx += 1
            state["next_start"], state["next_batch"] = start, batch_idx
            _save_progress(progress_path, state)  # commit point: batch is now "done"

            n_pos += len(positives)
            module_logger.info(f"Batch {batch_idx - 1}: {len(positives)}/{len(batch)} positive "
                               f"({start} seen, {n_pos} positives this run)")

        print(f"Done. {n_pos} positives written this run to {output_file}")


def run_filter(dataset, id_column, model_name, turn_len_limit, conv_turn_limit, output_file, resume,
               data_limit=None, batch_size=None, conversation_column="conversation"):
    """Keep only conversations about romantic relationship issues and write them to output_file (JSONL)."""

    if batch_size is None:
        batch_size = BATCH_SIZE
    if turn_len_limit is None:
        turn_len_limit = TURN_LEN_LIMIT
    if conv_turn_limit is None:
        conv_turn_limit = CONV_TURN_LIMIT

    key = "RelationshipIssuesRelated"
    system_prompt = f"""You are a helpful conversation analysis system. Your task is to determine whether a conversation involves the user discussing issues in their romantic relationship. Some examples of romantic relationship issues are: disagreements, conflicts, arguments, infidelity, jealousy, control, and communication problems. The user may be asking for advice, venting, or looking to understand their feelings or their partner's behaviour. Flag the conversation if it is related to romantic relationship issues and provide a brief reason for your decision. Respond with a single valid JSON object and nothing else. No markdown, no code fences, no explanation before or after. The JSON must have exactly these keys:
        {{
        "{key}": <boolean>,
        "Reasoning": <string, max 2 sentences>
        }}
        Rules:
            - Use double quotes for all keys and string values.
            - Do not add extra keys.
            - If you are unsure, still return the JSON with your best answer.
            - Do not include any text outside of the JSON object.
            - Do not include \\n, \\t, or any other escape characters in the JSON output.
        """

    try:
        runner = EvaluationRunner(model_name=model_name, key=key)
        print(f"Using model {model_name} on device {runner.device}.")
        dataset = dataset.select_columns([id_column, conversation_column])  # keep only the conversation column to save memory
        runner.label_and_filter(
            dataset=dataset, system_prompt=system_prompt, turn_len_limit=turn_len_limit,
            conv_turn_limit=conv_turn_limit, output_file=output_file, resume=resume,
            conversation_column=conversation_column, 
            data_limit=data_limit, batch_size=batch_size,
        )
    except Exception as e:
        module_logger.error(f"Error during filtering: {e}")
        raise

