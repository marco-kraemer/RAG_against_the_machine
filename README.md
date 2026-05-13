_This project has been created as part of the 42 curriculum by msantos2._

# RAG against the machine

## Description

This project implements a Retrieval-Augmented Generation (RAG) pipeline designed to answer questions over the `vLLM` codebase. It ingests source code and documentation, indexes it using the BM25 retrieval method, and utilizes a Local Large Language Model (Qwen/Qwen2.5-0.5B-Instruct) to generate grounded, source-cited natural language answers. It achieves robust recall@k metrics to validate its retrieval capabilities.

## Instructions

### Compilation/Installation

The project uses `uv` for modern, fast dependency management.

```bash
make install
```

Alternatively: `uv sync`

### Execution

To run the main search functionality on a single query:

```bash
make run
```

To index the repository (required before searching):

```bash
uv run python -m student index --max_chunk_size 2000
```

To run a CLI search query:

```bash
uv run python -m student search "How to configure OpenAI server?" -k 10
```

To answer a question using the LLM:

```bash
uv run python -m student answer "How to configure OpenAI server?" -k 5
```

To run bulk evaluation pipelines:

```bash
uv run python -m student search_dataset --dataset_path datasets_public/public/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results -k 10

uv run python -m student evaluate --student_search_results_path data/output/search_results/dataset_docs_public.json --dataset_path datasets_public/public/AnsweredQuestions/dataset_docs_public.json -k 10

uv run python -m student answer_dataset --student_search_results_path data/output/search_results/dataset_docs_public.json --save_directory data/output/search_results_and_answer
```

## Resources

- **BM25s**: Used for high-speed lexical search.
- **Transformers**: Employed to load and execute the Qwen model.
- **Pydantic**: Used for strong typing and data validation across the RAG pipeline models.
- **Fire**: Utilized for dynamic CLI creation.

## System Architecture

The system consists of three main modules:

1. **Indexer (`indexer.py`)**: Traverses the `vllm-0.10.1` directory, reading valid code and text files, chunks them based on configured maximum character limits, and uses the `bm25s` library to produce an inverted index.
2. **Retriever (`retriever.py`)**: Loads the BM25 index and corresponding file chunks. Given a query or dataset of queries, it tokenizes the query and retrieves the top-k overlapping chunks, mapped back to their original character offsets (`first_character_index`, `last_character_index`).
3. **Generator (`generator.py`)**: Consumes the context provided by the Retriever and crafts a prompt for the `Qwen` causal language model. It truncates the context to a maximum character/token length to fit the LLM context window and returns a synthesized, grounded answer.

## Chunking Strategy

The ingestion system applies a context-aware character limit approach. It initially attempts to split text into chunks based on double newlines (paragraphs/semantic blocks). If a logical block exceeds the defined limit (e.g., `2000` characters), it dynamically breaks the content down by single newlines, and as a last resort, hard character splits. It carefully tracks the substring offset within the original file to ensure character indices map perfectly to ground-truth validations.

## Retrieval Method

We utilized **BM25** (Best Matching 25) powered by the `bm25s` Python library. It offers rapid and exact lexical searching via Term Frequency-Inverse Document Frequency (TF-IDF) mechanics, optimized for high recall on specific code queries and terminologies standard in a framework codebase.

## Performance Analysis

The evaluation module computes `Recall@k` for k={1, 3, 5, 10}. The retrieval system successfully identifies relevant sources by ensuring at least 5% textual overlap. BM25 performs exceptionally well for exact keyword matching within code snippets (e.g., function names, variable names), providing a robust foundation for the generator.

## Design Decisions

- **`bm25s` over `scikit-learn`**: `bm25s` is written in Rust/C and explicitly designed for BM25 efficiency, scaling rapidly over large codebases like `vLLM` without significant memory overhead.
- **Offsets via `str.find`**: Extracting substring start and end indices using the core string module after chunk generation to avoid compounding offset errors common when iterating character arrays manually.
- **Modular CLI structure**: Each phase (index, search, evaluate, answer) is logically disjointed inside the `student` module and mapped directly to Fire methods, enabling seamless debugging and scalability.

## Challenges Faced

- **Index/Offset Mapping**: Correctly mapping character start and end indices back to the original source text during advanced chunking rules required careful iteration and validation using `.find()` mechanics to avoid zero-overlap errors.
- **Context Length Limitations**: Integrating local LLM context necessitated rigorous text truncation logic prior to injection into the prompt to avoid `CUDA Out of Memory` issues.

## Example Usage

```bash
$ uv run python -m student search "How to configure OpenAI server?" -k 2

{
    "question_id": "cli_query",
    "question": "How to configure OpenAI server?",
    "retrieved_sources": [
        {
            "file_path": "docs/source/serving/openai_compatible_server.rst",
            "first_character_index": 541,
            "last_character_index": 1289
        },
        ...
    ]
}
```

## Resources

- [Make your own RAG](https://huggingface.co/blog/ngxson/make-your-own-rag)
- [Understanding TF-IDF and BM-25](https://kmwllc.com/index.php/2020/03/20/understanding-tf-idf-and-bm-25/)
- [BM25S](https://bm25s.github.io/)
