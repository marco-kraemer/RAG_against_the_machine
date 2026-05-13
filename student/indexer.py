import json
import os
from pathlib import Path
from typing import List

import bm25s
from tqdm import tqdm


class ChunkIndexer:
    def __init__(self, repo_path="data/raw/vllm-0.10.1", index_dir="data/index"):
        self.repo_path = Path(repo_path)
        self.index_dir = Path(index_dir)

    #        self.supported_extensions = {
    #            ".py",
    #            ".md",
    #            ".rst",
    #            ".txt",
    #            ".c",
    #            ".cpp",
    #            ".h",
    #            ".cu",
    #            ".cuh",
    #        }

    def chunk_text(self, text, max_chunk_size) -> List[dict]:
        """
        Splits text into chunks based on double newlines, then single newlines,
        then hard character splits if necessary.
        """
        final_chunks: List[dict] = []

        # Step 1: Split by double newlines
        blocks: List[str] = text.split("\n\n")
        current_offset: int = 0

        for block in blocks:
            block_start: int = text.find(block, current_offset)
            if block_start == -1:
                block_start = current_offset

            if len(block) <= max_chunk_size:
                final_chunks.append(
                    {
                        "content": block,
                        "first_character_index": block_start,
                        "last_character_index": block_start + len(block),
                    }
                )
            else:
                # Step 2: Split by single newline
                sub_blocks: List[str] = block.split("\n")
                sub_offset: int = block_start
                for sub_block in sub_blocks:
                    actual_sub_start: int = text.find(sub_block, sub_offset)
                    if actual_sub_start == -1:
                        actual_sub_start = sub_offset

                    if len(sub_block) <= max_chunk_size:
                        if sub_block.strip():  # Skip empty sub-blocks
                            final_chunks.append(
                                {
                                    "content": sub_block,
                                    "first_character_index": actual_sub_start,
                                    "last_character_index": actual_sub_start
                                    + len(sub_block),
                                }
                            )
                    else:
                        # Step 3: Hard character split
                        for i in range(0, len(sub_block), max_chunk_size):
                            chunk_content = sub_block[i : i + max_chunk_size]
                            final_chunks.append(
                                {
                                    "content": chunk_content,
                                    "first_character_index": actual_sub_start + i,
                                    "last_character_index": actual_sub_start
                                    + i
                                    + len(chunk_content),
                                }
                            )
                    sub_offset = actual_sub_start + len(sub_block)

            current_offset = block_start + len(block)

        return final_chunks

    def index_repository(self, max_chunk_size=2000) -> None:
        print(f"Indexing repository at {self.repo_path}...")

        # 1. Walk through files
        files_to_index: List[Path] = []
        for root, _, files in os.walk(self.repo_path):
            # Skip hidden directories
            # dirs[:] = [d for d in dirs if not d.startswith(".")]
            for file in files:
                # if any(file.endswith(ext) for ext in self.supported_extensions):
                files_to_index.append(Path(root) / file)

        print(f"Found {len(files_to_index)} files. Chunking...")

        # 2. Extract chunks
        all_chunks: List[dict] = []
        for file_path in tqdm(files_to_index):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                relative_path = os.path.relpath(file_path, self.repo_path)
                file_chunks = self.chunk_text(content, max_chunk_size)

                for chunk in file_chunks:
                    chunk["file_path"] = relative_path
                    all_chunks.append(chunk)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        if not all_chunks:
            print("No chunks found to index.")
            return

        print(f"Total chunks created: {len(all_chunks)}. Building BM25 index...")

        # 3. Create BM25 index
        corpus: List = [c["content"] for c in all_chunks]

        # Simple tokenization for BM25S
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en")
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        # 4. Save index and metadata
        self.index_dir.mkdir(parents=True, exist_ok=True)
        retriever.save(self.index_dir / "bm25_model", corpus=corpus)

        # Save chunk metadata for retrieval mapping
        with open(self.index_dir / "chunks_metadata.json", "w") as f:
            json.dump(all_chunks, f, indent=2)

        print(f"Indexing complete. Index saved to {self.index_dir}")
