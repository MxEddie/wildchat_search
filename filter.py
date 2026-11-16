"""
Batched conversation evaluation with a local Hugging Face model.

Takes a Hugging Face ``Dataset`` (or a streaming ``IterableDataset``) of conversations,
asks a local text-generation model to label each one, and returns a new ``Dataset``
containing only the conversations that match.

Instead of calling the pipeline once per conversation, rows are read in chunks, sorted by
prompt length (to minimise padding) and sent to the model in real batches.
"""

import json
import logging
import os
from itertools import islice
from time import sleep

import transformers
from datasets import Dataset, IterableDataset
from tqdm import tqdm

from utils import parse_dict, sample_output

module_logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 128  # rows generated, written and fsynced together (the checkpoint unit)


def conversation_to_turns(conversation) -> list[str]:
    """Normalise a dataset conversation into a list of turn strings.

    Accepts a list of messages ({"role": ..., "content": ...}, as in WildChat / LMSYS-Chat),
    a list of plain strings, or the older {"turn_0": text, ...} dict.
    """
    if isinstance(conversation, dict):
        return [str(t) for t in conversation.values()]
    return [str(t["content"]) if isinstance(t, dict) else str(t) for t in conversation]


class EvaluateModel:
    """
    Evaluates conversations from a Hugging Face dataset with a local text-generation model.

    Attributes:
        model (str): Model name or path.
        retry_attempts (int): Retries for a single prompt that keeps failing.
        batch_size (int): Number of prompts sent to the model at once (tune to your VRAM).
        max_output_tokens (int | None): Max new tokens per response.
        generation_kwargs (dict): Extra generation kwargs (do_sample defaults to False).
    """

    def __init__(self, model, hf_access_token=None, retry_attempts=3, batch_size=16,
                 max_output_tokens=128, pipeline_kwargs=None, **generation_kwargs):
        self.model = model
        self.retry_attempts = int(retry_attempts)
        self.batch_size = int(batch_size)
        self.max_output_tokens = max_output_tokens
        self.generation_kwargs = {"do_sample": False, **generation_kwargs}
        if max_output_tokens is not None:
            self.generation_kwargs["max_new_tokens"] = max_output_tokens

        # pipeline_kwargs lets you pass e.g. device_map="auto" or dtype="auto"
        self.client = transformers.pipeline(
            "text-generation", model=model, token=hf_access_token, **(pipeline_kwargs or {})
        )
        tokenizer = self.client.tokenizer
        tokenizer.padding_side = "left"  # required for batched decoder-only generation
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------ prompts

    def _build_prompt(self, conversation, system_prompt, key, conv_turn_limit, turn_len_limit) -> str:
        turns = conversation_to_turns(conversation)[:conv_turn_limit]
        turns = {f"turn_{i}": text[:turn_len_limit] for i, text in enumerate(turns)}
        user_message = (
            "Evaluate the following conversation. Respond with a single valid JSON object and nothing else. "
            "No markdown, no code fences, no explanation before or after. "
            "The JSON must have exactly these keys:\n"
            "{\n"
            f'"{key}": <boolean>,\n'
            '"Reasoning": <string, max 2 sentences>\n'
            "}\n"
            f"Conversation: {turns}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        return self.client.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    # --------------------------------------------------------------- generation

    def _run(self, prompts: list[str]) -> list[str]:
        outputs = self.client(
            prompts, batch_size=len(prompts), return_full_text=False, **self.generation_kwargs
        )
        return [(o[0] if isinstance(o, list) else o)["generated_text"] for o in outputs]

    def _generate_batch(self, prompts: list[str]) -> list[str | None]:
        """Run one batch. On failure (e.g. OOM) split it in half; a lone prompt is retried, then skipped."""
        if len(prompts) == 1:
            for attempt in range(self.retry_attempts):
                try:
                    return self._run(prompts)
                except Exception as e:
                    module_logger.error(
                        f"Generation failed (attempt {attempt + 1}/{self.retry_attempts}): {e}"
                    )
                    sleep(2 ** attempt)
            return [None]
        try:
            return self._run(prompts)
        except Exception as e:
            module_logger.warning(f"Batch of {len(prompts)} failed ({e}); splitting it in half.")
            mid = len(prompts) // 2
            return self._generate_batch(prompts[:mid]) + self._generate_batch(prompts[mid:])

    def _generate_all(self, prompts: list[str]) -> list[str | None]:
        """Generate for many prompts: sort by length, run in batches, restore original order."""
        order = sorted(range(len(prompts)), key=lambda j: len(prompts[j]))
        sorted_outputs: list[str | None] = []
        for start in range(0, len(order), self.batch_size):
            batch = [prompts[j] for j in order[start:start + self.batch_size]]
            sorted_outputs.extend(self._generate_batch(batch))
        results: list[str | None] = [None] * len(prompts)
        for pos, j in enumerate(order):
            results[j] = sorted_outputs[pos]
        return results

    @staticmethod
    def _parse(text, key):
        """Turn raw model text into {key: bool, "Reasoning": str}, or None if unusable."""
        if text is None:
            return None
        try:
            parsed = parse_dict(text)
            value = parsed[key]
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered not in ("true", "false"):
                    raise ValueError(f"non-boolean value {value!r}")
                value = lowered == "true"
            return {key: bool(value), "Reasoning": str(parsed.get("Reasoning", ""))}
        except Exception as e:
            module_logger.warning(f"Could not parse model response ({e}): {text[:200]!r}")
            return None

    # ------------------------------------------------------------- checkpointing

    @staticmethod
    def _load_progress(path: str) -> list[dict]:
        """Read the JSONL progress file, dropping (and truncating away) a partially written last line."""
        records, good_bytes = [], 0
        if not os.path.exists(path):
            return records
        with open(path, "rb") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    break
                good_bytes += len(line)
        with open(path, "r+b") as f:
            f.truncate(good_bytes)
        return records

    # ----------------------------------------------------------------- main API

    def evaluate(self, dataset, system_prompt, key, output_file,
                 conversation_column="conversation", id_column=None, keep_value=True,
                 conv_turn_limit=None, turn_len_limit=None, chunk_size=DEFAULT_CHUNK_SIZE,
                 resume=False, data_limit=None) -> Dataset:
        """
        Label every conversation in ``dataset`` and return a Dataset of the rows to keep.

        Parameters:
            dataset: datasets.Dataset or datasets.IterableDataset (e.g. load_dataset(..., streaming=True)).
            system_prompt (str): System prompt describing what to flag.
            key (str): Boolean key the model must return, e.g. "IssuesRelated".
            output_file (str): Path prefix. Progress is appended to "<output_file>.<key>.progress.jsonl".
            conversation_column (str): Column holding the conversation.
            id_column (str | None): Column holding a unique id (falls back to the row index).
            keep_value (bool | None): Keep rows whose evaluation[key] equals this. Rows the model
                failed to label are dropped. None keeps every row (just adds the "Evaluation" column).
                The progress file only stores rows that pass this filter, so reuse the same value on resume.
            conv_turn_limit / turn_len_limit: Max turns per conversation / max characters per turn.
            chunk_size (int): Rows per checkpoint. batch_size (set in __init__) controls GPU batching.
            resume (bool): Continue from the progress file instead of starting over.
            data_limit (int | None): Only process the first N rows.

        Returns:
            Dataset: the kept rows, each with an added "Evaluation" column.
        """
        if system_prompt is None:
            raise ValueError("A system_prompt is required.")
        if self.max_output_tokens is not None:
            system_prompt = f"{system_prompt} Limit your response to {self.max_output_tokens} tokens."

        streaming = isinstance(dataset, IterableDataset)
        if data_limit is not None:
            dataset = dataset.take(data_limit) if streaming else dataset.select(range(min(data_limit, len(dataset))))
        total = None if streaming else len(dataset)

        progress_path = f"{output_file}.{key}.progress.jsonl"
        if resume:
            records = self._load_progress(progress_path)
        else:
            records = []
            if os.path.exists(progress_path):
                os.remove(progress_path)
        start = len(records)
        if start:
            module_logger.info(f"Resuming from row {start}.")

        remaining = dataset.skip(start) if streaming else dataset.select(range(start, len(dataset)))
        rows = iter(remaining)
        idx = start

        with open(progress_path, "a") as out, \
                tqdm(total=total, initial=start, desc=f"Evaluating {key}", unit="conv") as pbar:
            while True:
                chunk = list(islice(rows, chunk_size))
                if not chunk:
                    break
                prompts = [
                    self._build_prompt(row[conversation_column], system_prompt, key, conv_turn_limit, turn_len_limit)
                    for row in chunk
                ]
                responses = self._generate_all(prompts)

                for row, response in zip(chunk, responses):
                    evaluation = self._parse(response, key)
                    keep = keep_value is None or (evaluation is not None and evaluation[key] == keep_value)
                    record = {
                        "idx": idx,
                        "id": row.get(id_column, idx) if id_column else idx,
                        "Evaluation": evaluation,
                        "row": {**row, "Evaluation": evaluation} if keep else None,
                    }
                    records.append(record)
                    out.write(json.dumps(record, default=str) + "\n")  # default=str handles datetimes
                    idx += 1

                out.flush()
                os.fsync(out.fileno())
                pbar.update(len(chunk))

        kept = [r["row"] for r in records if r["row"] is not None]
        module_logger.info(f"Evaluated {len(records)} conversations; kept {len(kept)}.")
        if not kept:
            raise ValueError("No conversations found after filtering.")
        return Dataset.from_list(kept)


def filter_conversations(dataset, model_name: str = "Qwen/Qwen3-0.6B", system_prompt: str = None,
                         key: str = "Sexual_Content", keep_value: bool = True,
                         turn_len_limit: int = None, conv_turn_limit: int = None,
                         conversation_column: str = "conversation", id_column: str = None,
                         batch_size: int = 16, resume: bool = False, output_file: str = "filter_output",
                         data_limit: int = None) -> Dataset:
    model = EvaluateModel(model=model_name, hf_access_token=os.environ.get("HF_TOKEN"),
                          retry_attempts=3, batch_size=batch_size)
    return model.evaluate(
        dataset, system_prompt=system_prompt, key=key, output_file=output_file,
        conversation_column=conversation_column, id_column=id_column, keep_value=keep_value,
        conv_turn_limit=conv_turn_limit, turn_len_limit=turn_len_limit,
        resume=resume, data_limit=data_limit,
    )


def run_filter(dataset, model_name, turn_len_limit, conv_turn_limit, output_file, resume,
               data_limit=None, batch_size=16, conversation_column="conversation", id_column=None):
    """Keep only conversations about romantic relationship issues and write them to ``output_file`` (JSONL)."""
    key = "IssuesRelated"
    system_prompt = f"""You are a helpful conversation analysis system. Your task is to determine whether a conversation involves someone discussing issues in their romantic relationship. Some examples of romantic relationship issues are: disagreements, conflicts, arguments, infidelity, jealousy, control, and communication problems. The user may be asking for advice, venting, or looking to understand their feelings or their partner's behaviour. Flag the conversation if it is related to romantic relationship issues and provide a brief reason for your decision. Respond with a single valid JSON object and nothing else. No markdown, no code fences, no explanation before or after. The JSON must have exactly these keys:
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
        filtered = filter_conversations(
            dataset, model_name=model_name, system_prompt=system_prompt, key=key, keep_value=True,
            turn_len_limit=turn_len_limit, conv_turn_limit=conv_turn_limit,
            conversation_column=conversation_column, id_column=id_column,
            batch_size=batch_size, resume=resume, output_file=output_file, data_limit=data_limit,
        )
        filtered.to_json(output_file)  # JSON Lines
        module_logger.info(f"Filtered conversations saved to {output_file}")

        # sample_output expects the old {id: {"Conversation": {...}, "Evaluation": {...}}} layout
        legacy = {
            (row[id_column] if id_column else i): {
                "Conversation": {f"turn_{j}": t for j, t in enumerate(conversation_to_turns(row[conversation_column]))},
                "Evaluation": row["Evaluation"],
            }
            for i, row in enumerate(filtered)
        }
        sample_ids, sample_conversations = sample_output(legacy, sample_size=5, turn_count_max=2, label=key)
        print(f"IDs of filtered conversations: {sample_ids}")
        print(f"Sample of filtered conversations: {sample_conversations}")
    except ValueError as e:
        module_logger.error(f"Filtering failed: {e}")
        print(f"Filtering failed: {e}")
        open(output_file, "w").close()