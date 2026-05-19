from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from student.models import MinimalAnswer, MinimalSource

_tokenizer = None
_model = None


def get_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        model_name = "Qwen/Qwen3-0.6B"
        _tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=False,
        )

        model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=model_dtype,
            device_map="auto",
        )
    return _tokenizer, _model


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
        block = f"{header}\n{chunk['content']}".strip()
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
        "If the context does not contain enough information, say that the "
        "context does not contain enough information to answer.\n"
        "Keep the answer concise and self-contained.\n"
        " Don't repeat yourself.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


def clean_generated_answer(answer: str) -> str:
    """Clean model-only generated text without inventing citations."""
    cleaned = answer.strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[-1].strip()
    if cleaned.startswith("Answer:"):
        cleaned = cleaned[len("Answer:") :].strip()
    if cleaned.find("."):
        cleaned = cleaned.split(".", 1)[0].strip() + "."
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
        max_new_tokens=256,
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
