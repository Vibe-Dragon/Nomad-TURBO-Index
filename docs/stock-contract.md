# Stock contract


Turbo stores vectors stock NOMAD can search. Do not change these if you port the indexer:

- Model `nomic-embed-text:v1.5`, 768 dimensions, cosine distance, collection `nomad_knowledge_base`.
- The embedder sees `search_document: ` plus the chunk. The stored payload does not include that prefix. NOMAD queries with `search_query: `.
- `source` is the container path, such as `/app/storage/zim/some.zim`.

```python
def embed_input(chunk: str, prefix: str = DOCUMENT_PREFIX) -> tuple[str, str]:
    """Return (prefixed model input, payload text without the prefix).

    The model sees `search_document: ` plus the chunk, which is the stock
    NOMAD document prefix. The text stored in Qdrant omits that prefix.
    NOMAD queries with `search_query: ` through its own Ollama. Turbo does
    not change either prefix.
    """
```
