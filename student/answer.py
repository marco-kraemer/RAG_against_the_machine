import sys
import os
import logging
from student.search import search
from transformers import pipeline, logging as transformers_logging
import torch
from typing import List, Dict

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
transformers_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

_generator = None


def get_generator():
    global _generator
    if _generator is None:
        model_name = "Qwen/Qwen3-0.6B"
        _generator = pipeline(
            "text-generation",
            model=model_name,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
    return _generator


def rag(query: str, k: int):
    generator = get_generator()

    search_data: List[Dict] = search(query, k)

    retrieved_context: str = ""

    for data in search_data:
        file_path = data["file_path"]
        first_char = data["first_character_index"]
        last_char = data["last_character_index"]

        try:
            with open(f"./data/raw/vllm-0.10.1/{file_path}", "r") as f:
                content = f.read()
                chunk_content = content[first_char:last_char]
                retrieved_context += f"Source {file_path}:\n{chunk_content}\n---\n"
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    prompt = f"Use the following context to answer the question.\n\nContext:\n{retrieved_context}\n\nQuestion: {query}\n\nAnswer:"

    messages = [
        {
            "role": "system",
            "content": "You are a vLLM expert. Answer the question in one sentence.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    output = generator(
        messages,
        max_new_tokens=100,
        truncation=True,
        do_sample=False,
    )

    full_text = output[0]["generated_text"]

    # Extract the assistant's response from the chat format
    if isinstance(full_text, list):
        answer = full_text[-1]["content"]
    else:
        # Fallback for non-chat format
        answer = full_text

    # Remove thinking block if present
    if "<think>" in answer:
        if "</think>" in answer:
            answer = answer.split("</think>")[-1].strip()
        else:
            parts = answer.split("<think>")
            if len(parts) > 1:
                answer = parts[-1].split("</think>")[-1].strip()

    print(answer)
    return answer
