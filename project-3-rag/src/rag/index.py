"""Stage 2 — Index: chunk the ATT&CK documents, embed them, build a FAISS store.

Loads documents.jsonl, splits into overlapping chunks, embeds with a
sentence-transformers bi-encoder, and persists a FAISS vector store that the
retriever loads at query time. Metadata (technique id, name, url) rides along on
each chunk for citation.
"""

from __future__ import annotations

import json

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import load_config, resolve


def build_embeddings(cfg) -> HuggingFaceEmbeddings:
    e = cfg["embeddings"]
    return HuggingFaceEmbeddings(
        model_name=e["model"],
        model_kwargs={"device": e.get("device", "cpu")},
        encode_kwargs={"normalize_embeddings": e.get("normalize", True)},
    )


def load_documents(cfg) -> list[Document]:
    path = resolve(cfg["paths"]["documents"])
    docs = []
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            docs.append(Document(
                page_content=rec["text"],
                metadata={
                    "technique_id": rec["technique_id"],
                    "name": rec["name"],
                    "tactics": ", ".join(rec.get("tactics", [])),
                    "url": rec.get("url", ""),
                    "is_subtechnique": rec.get("is_subtechnique", False),
                },
            ))
    return docs


def chunk(docs: list[Document], cfg) -> list[Document]:
    c = cfg["chunking"]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=c["chunk_size"],
        chunk_overlap=c["chunk_overlap"],
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def main() -> int:
    cfg = load_config()
    docs = load_documents(cfg)
    chunks = chunk(docs, cfg)
    print(f"[index] {len(docs)} docs -> {len(chunks)} chunks "
          f"(size={cfg['chunking']['chunk_size']}, overlap={cfg['chunking']['chunk_overlap']})")

    embeddings = build_embeddings(cfg)
    print(f"[index] embedding with {cfg['embeddings']['model']} "
          f"on {cfg['embeddings'].get('device', 'cpu')} ...")
    store = FAISS.from_documents(chunks, embeddings)

    out = resolve(cfg["paths"]["faiss_index"])
    out.parent.mkdir(parents=True, exist_ok=True)
    store.save_local(str(out))
    print(f"[index] saved FAISS store ({store.index.ntotal} vectors) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
