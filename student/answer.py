import logging
import os
import re
from typing import Dict, List

import torch
from student.search import search
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from transformers import logging as transformers_logging

# Suppress warnings and noisy logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"

transformers_logging.set_verbosity_error()

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

_generator = None


def get_generator():
    """
    Lazily load and cache the text-generation pipeline.
    """
    global _generator
    return _generator


def build_context(search_data: List[Dict]) -> str:
    """
    Build retrieval context from retrieved chunks.
    """
    retrieved_context = ""

    for data in search_data:
        file_path = data["file_path"]
        first_char = data["first_character_index"]
        last_char = data["last_character_index"]

        full_path = f"./data/raw/vllm-0.10.1/{file_path}"

        try:
            with open(full_path, "r", encoding="utf-8") as file:
                file_content = file.read()

                chunk_content = file_content[first_char:last_char]

                retrieved_context += (
                    f"\n=== SOURCE: {file_path} ===\n" f"{chunk_content}\n"
                )

        except Exception as error:
            print(f"Error reading {file_path}: {error}")
            continue

    return retrieved_context.strip()


def rag(query: str, k: int = 5) -> str:
    """
    Execute the RAG pipeline:
    1. Retrieve relevant chunks
    2. Build context
    3. Generate grounded answer
    """

    model_name = "Qwen/Qwen3-0.6B"

    tokernizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    search_data: List[Dict] = search(query, k)

    retrieved_context = build_context(search_data)

    prompt = f"""You are a vLLM expert.\n
Answer ONLY using the provided context.\n
Answer it one sentence.\n
Context:\n{retrieved_context}\n\n
Question:\n{query}
Answer:"""

    inputs = tokernizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )

    answer = tokernizer.decode(
        output[0],
        skip_special_tokens=True,
    )

    print(answer)

    return answer
