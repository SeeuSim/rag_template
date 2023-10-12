from functools import partial
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from pathlib import Path
from ray.data import ActorPoolStrategy, from_items

import numpy as np
import os
import pdfplumber
import re

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

EMBEDDING_DIMENSIONS = {
    "thenlper/gte-base": 768,
    "thenlper/gte-large": 1024,
    "BAAI/bge-large-en": 1024,
    "text-embedding-ada-002": 1536,
}


def strip_page_num(text):
    p_size = len("1000")
    for idx in range(p_size, 1, -1):
        if text[-idx] == "\n":
            return text[:-idx]
    return text


def extract_sections(record):
    # Define your regex pattern to match section headings
    # `\n2.1.1 Amazing Grace\n` and
    # NOT `2.1.1 afff addfs . . . 234`
    # NOR `section. 2.1.1. indicates \n`
    pattern = r"[^\s]+(\d+\.){1,}(\d)(\s+[^\.]{2,})+\n"

    with pdfplumber.open(record["path"]) as pdf:
        sections = []

        v = [p for p in pdf.pages]

        for page in v[17:]:
            text = page.extract_text()

            matches = re.finditer(pattern, text)

            v = [match for match in matches]

            if len(v) == 0 and sections:
                sections[-1]["Chunks"].append(strip_page_num(text))
                continue

            for idx, match in enumerate(v):
                section_start = match.start()
                if sections and idx == 0:
                    sections[-1]["Chunks"].append(text[0:section_start])
                if idx < len(v) - 1:
                    section_end = v[idx + 1].start()
                    chunk = text[section_start:section_end].strip()
                else:
                    chunk = strip_page_num(text[section_start : len(text)].strip())
                sections.append({"Source": "#" + match.group(), "Chunks": [chunk]})

        pdf.close()

        return [
            {"source": item["Source"], "text": "\n".join(item["Chunks"])}
            for item in sections
        ]


def get_document_sections():
    DOCS_DIR = Path(
        "./documents/",
    )
    ds = from_items(
        [{"path": path} for path in DOCS_DIR.rglob("*.pdf") if not path.is_dir()]
    )
    sections_ds = ds.flat_map(extract_sections)

    return sections_ds


def chunk_section(section, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.create_documents(
        texts=[section["text"]], metadatas=[{"source": section["source"]}]
    )
    return [
        {"text": chunk.page_content, "source": chunk.metadata["source"]}
        for chunk in chunks
    ]


class EmbedChunks:
    def __init__(self, model_name):
        if model_name == "text-embedding-ada-002":
            self.embedding_model = OpenAIEmbeddings(
                model=model_name,
                openai_api_base=os.environ["OPENAI_API_BASE"],
                openai_api_key=os.environ["OPENAI_API_KEY"],
            )
        else:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                # model_kwargs={"device": "cuda"},
                # encode_kwargs={"device": "cuda", "batch_size": 100},
                encode_kwargs={"batch_size": 100},
            )

    def __call__(self, batch):
        embeddings = self.embedding_model.embed_documents(batch["text"])
        return {
            "text": batch["text"],
            "source": batch["source"],
            "embeddings": embeddings,
        }


if __name__ == "__main__":
    sections_ds = get_document_sections()
    chunks_ds = sections_ds.flat_map(
        partial(chunk_section, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    )

    embedding_model_name = "thenlper/gte-base"
    embedded_chunks = chunks_ds.map_batches(
        EmbedChunks,
        fn_constructor_kwargs={"model_name": embedding_model_name},
        batch_size=100,
        # num_gpus=1,
        compute=ActorPoolStrategy(size=2),
    )

    print(embedded_chunks.take(1))
