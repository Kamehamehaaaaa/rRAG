# Pluggable Embedding based RAG

FAISS Index based retrieval.

Files supported as Input:
- Text files
- PDFs
- PPTs

Embeddings Supported:
- Sentence Transformer Embedding
- OpenAI embedding
- Add your own embedding (Currently by cloning and calling embedding.registry.register())

FAISS Indexed Vector DB store
- uses manhattan distance

APIs:
- POST /query
{
    "request": "user request text"
}

200 OK
{
    "results": [{
        "docId":
        "chunkId":
        "text":
        "distance":
    }]
}

500 Internal Server Error

Command Line:
python -m ingest.run_ingest <build/add> --data-dir <dir> --embedding <sentence_transformer/openai> --chunk-size <3>