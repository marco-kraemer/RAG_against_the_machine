import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
from tqdm import tqdm

from student.models import (
    MinimalAnswer,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa

_tokenizer: Optional[Any] = None
_model: Optional[Any] = None
RAW_REPO_PATH = Path("data/raw/vllm-0.10.1")
RAW_REPO_PREFIX = "data/raw/vllm-0.10.1/"
MODEL_NAME = "Qwen/Qwen3-0.6B"


def _offline_requested() -> bool:
    """Return whether the user explicitly requested offline model loading."""
    return (
        os.environ.get("HF_HUB_OFFLINE") == "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    )


def _load_model(local_files_only: bool) -> Tuple[Any, Any]:
    """Load tokenizer and model with the requested
    Hugging Face cache policy."""
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=local_files_only,
    )
    model_dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        local_files_only=local_files_only,
        dtype=model_dtype,
        device_map="cpu",
    )
    return tokenizer, model


def get_model() -> Tuple[Any, Any]:
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        try:
            _tokenizer, _model = _load_model(local_files_only=True)
        except Exception as local_error:
            if _offline_requested():
                raise RuntimeError(
                    "Could not load Qwen/Qwen3-0.6B from the local "
                    "Hugging Face cache. Cache the model once before "
                    "running answer generation offline."
                ) from local_error
            try:
                _tokenizer, _model = _load_model(local_files_only=False)
            except Exception as remote_error:
                raise RuntimeError(
                    "Could not load Qwen/Qwen3-0.6B from Hugging Face "
                    "cache or download it from Hugging Face."
                ) from remote_error
    assert _tokenizer is not None and _model is not None
    return _tokenizer, _model


def chunks_to_sources(chunks: List[Dict[str, Any]]) -> List[MinimalSource]:
    """Convert retrieved chunks to subject-compliant source metadata."""
    return [
        MinimalSource(
            file_path=public_source_path(str(chunk["file_path"])),
            first_character_index=chunk["first_character_index"],
            last_character_index=chunk["last_character_index"],
        )
        for chunk in chunks
    ]


def public_source_path(file_path: str) -> str:
    """Return the source path shape expected by the moulinette datasets."""
    normalized = file_path.replace("\\", "/")
    if normalized.startswith(RAW_REPO_PREFIX):
        return normalized
    return f"{RAW_REPO_PREFIX}{normalized}"


def source_to_chunk(source: MinimalSource) -> Dict[str, Any]:
    """Convert a public source reference into an internal context chunk."""
    return {
        "file_path": source.file_path,
        "first_character_index": source.first_character_index,
        "last_character_index": source.last_character_index,
        "content": "",
    }


def sources_to_chunks(sources: List[MinimalSource]) -> List[Dict[str, Any]]:
    """Convert public source references into context chunks."""
    return [source_to_chunk(source) for source in sources]


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
    max_sources: int = 3,
) -> str:
    """Build source-labeled context from the first retrieved sources."""
    context_parts: List[str] = []

    for index, chunk in enumerate(chunks[:max_sources], start=1):
        header = f"[source {index}] {chunk['file_path']}"
        block = f"{header}\n{read_expanded_source(chunk)}".strip()
        context_parts.append(block)

    return "\n\n".join(context_parts)


def build_prompt(query: str, context: str) -> str:
    """Build the grounded generation prompt for Qwen."""
    return (
        "Answer the question about vLLM using only the context below. "
        "Be concise, self-contained, faithful to the sources. "
        "If the context is insufficient, say so. /no-think\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


def clean_generated_answer(answer: str) -> str:
    """Clean model-only generated text without inventing citations."""
    cleaned = answer.strip()
    if not cleaned:
        return cleaned

    for marker in ("Question:", "Context:", "Answer:"):
        marker_index = cleaned.find(marker)
        if marker_index != -1:
            cleaned = cleaned[:marker_index].strip()

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return " ".join(lines)


def generate_answer(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Generate a grounded answer from retrieved chunks."""
    context = build_context(chunks)
    if not context:
        return "I could not find relevant context to answer the question."

    prompt = build_prompt(query, context)
    tokenizer, model = get_model()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
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


def answer_dataset(
    student_search_results_path: str,
    save_directory: str,
) -> Path:
    """Generate answers for saved search results and write subject JSON."""
    input_path = Path(student_search_results_path)
    output_dir = Path(save_directory)
    output_path = output_dir / input_path.name

    try:
        with input_path.open("r", encoding="utf-8") as results_file:
            search_results = StudentSearchResults.model_validate_json(
                results_file.read()
            )
    except Exception as e:
        print(f"Error loading results {student_search_results_path}: {e}")
        raise SystemExit(1) from e

    print(
        "Loaded "
        f"{len(search_results.search_results)} questions from "
        f"{student_search_results_path}"
    )

    answers: List[MinimalAnswer] = []
    try:
        for result in tqdm(
            search_results.search_results,
            desc="Answering questions",
        ):
            chunks = sources_to_chunks(result.retrieved_sources)
            if chunks:
                answer = generate_answer(result.question, chunks)
            else:
                answer = "I could not find relevant context."
            answers.append(
                MinimalAnswer(
                    question_id=result.question_id,
                    question=result.question,
                    retrieved_sources=result.retrieved_sources,
                    answer=answer,
                )
            )
    except RuntimeError as e:
        print(f"Error generating answers: {e}")
        raise SystemExit(1) from e

    output = StudentSearchResultsAndAnswer(
        search_results=answers,
        k=search_results.k,
    )

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output_file:
            output_file.write(output.model_dump_json(indent=2))
            output_file.write("\n")
    except Exception as e:
        print(f"Error saving answers to {output_path}: {e}")
        raise SystemExit(1) from e

    print(f"Saved student_search_results_and_answer to {output_path}")
    return output_path
