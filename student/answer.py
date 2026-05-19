import os
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from student.models import MinimalAnswer, MinimalSource

_tokenizer = None
_model = None
RAW_REPO_PATH = Path("data/raw/vllm-0.10.1")


def get_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        model_name = "Qwen/Qwen3-0.6B"
        try:
            _tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=True,
            )

            model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            _model = AutoModelForCausalLM.from_pretrained(
                model_name,
                local_files_only=True,
                dtype=model_dtype,
                device_map="auto",
            )
        except Exception as e:
            raise RuntimeError(
                "Could not load Qwen/Qwen3-0.6B from the local Hugging Face "
                "cache. Cache the model once before running answer generation "
                "offline."
            ) from e
    return _tokenizer, _model


def warmup_answer() -> None:
    """Preload search and model resources before processing many answers."""
    from student.search import warmup_search

    warmup_search()
    get_model()


def chunks_to_sources(chunks: List[Dict[str, Any]]) -> List[MinimalSource]:
    """Convert retrieved chunks to subject-compliant source metadata."""
    return [
        MinimalSource(
            file_path=chunk["file_path"],
            first_character_index=chunk["first_character_index"],
            last_character_index=chunk["last_character_index"],
        )
        for chunk in chunks
    ]


def read_expanded_source(
    chunk: Dict[str, Any],
    max_chars: int = 1800,
) -> str:
    """Read a useful context window around a retrieved chunk."""
    file_path = Path(str(chunk["file_path"]))
    candidates = [file_path]
    if not file_path.is_absolute():
        candidates.append(RAW_REPO_PATH / file_path)

    content = str(chunk.get("content", ""))
    for candidate in candidates:
        try:
            text = candidate.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        start = max(0, int(chunk["first_character_index"]))
        end = min(len(text), int(chunk["last_character_index"]))
        window_end = min(len(text), max(end, start + max_chars))
        expanded = text[start:window_end].strip()
        if expanded:
            return expanded

    return content.strip()


def build_context(
    chunks: List[Dict[str, Any]],
    max_context_chars: int = 6000,
) -> str:
    """Build a bounded, source-labeled context from retrieved chunks."""
    context_parts: List[str] = []
    current_size = 0

    for index, chunk in enumerate(chunks, start=1):
        header = (
            f"[source {index}] "
            f"{chunk['file_path']}:"
            f"{chunk['first_character_index']}-"
            f"{chunk['last_character_index']}"
        )
        block = f"{header}\n{read_expanded_source(chunk)}".strip()
        separator_size = 2 if context_parts else 0
        remaining_chars = max_context_chars - current_size - separator_size

        if remaining_chars <= 0:
            break
        if len(block) > remaining_chars:
            if remaining_chars <= len(header) + 1:
                break
            block = block[:remaining_chars].rstrip()

        context_parts.append(block)
        current_size += len(block) + separator_size

    return "\n\n".join(context_parts)


def build_prompt(query: str, context: str) -> str:
    """Build the grounded generation prompt for Qwen."""
    return (
        "You are answering questions about the vLLM codebase.\n"
        "Use the provided context.\n"
        "Keep the answer very concise, you must answer in one sentence.\n"
        "Do not repeat yourself. /no-think\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


def clean_generated_answer(answer: str) -> str:
    """Clean model-only generated text without inventing citations."""
    cleaned = answer.strip()
    if cleaned.find("."):
        cleaned = cleaned.split(".")[0].strip() + "."
    elif cleaned.find("\n"):
        cleaned = cleaned.split("\n")[0].strip()
    return cleaned


def generate_answer(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Generate a grounded answer from retrieved chunks."""
    context = build_context(chunks)
    if not context:
        return "I could not find relevant context to answer the question."

    prompt = build_prompt(query, context)
    tokenizer, model = get_model()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=96,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_length = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[0][prompt_length:]
    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )
    return clean_generated_answer(generated_text)


def answer_query(query: str, k: int = 10) -> MinimalAnswer:
    """Retrieve context and return a structured answer for one query."""
    from student.search import _retrieve_chunks

    chunks = _retrieve_chunks(query, k)
    sources = chunks_to_sources(chunks)
    if not chunks:
        answer = "I could not find relevant context to answer the question."
    else:
        answer = generate_answer(query, chunks)

    return MinimalAnswer(
        question_id="cli_query",
        question=query,
        retrieved_sources=sources,
        answer=answer,
    )


def rag(query: str, k: int = 10) -> str:
    return answer_query(query, k).answer
