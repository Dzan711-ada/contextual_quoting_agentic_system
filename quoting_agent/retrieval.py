"""Qwen embeddings and a persistent Chroma collection with content-based caching."""

import hashlib
import json
import logging

from langchain_openai import OpenAIEmbeddings

from .catalog import Catalog
from .config import Settings


logger = logging.getLogger(__name__)


class CategoryRetriever:
    """Build vectors lazily; unchanged records are embedded only once."""

    def __init__(self, settings: Settings, catalog: Catalog):
        self.settings = settings
        self.catalog = catalog
        self._collection = None
        self._embeddings = None

    def prepare(self) -> int:
        """Persist complete rows, never arbitrary 150-character fragments."""
        if self._collection is not None:
            return self._collection.count()
        self.settings.require_embeddings()
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        texts = [record.text for record in self.catalog.records]
        signature = json.dumps([
            self.settings.embedding_model, self.settings.qwen_base_url, texts,
        ], ensure_ascii=False).encode("utf-8")
        name = "categories-" + hashlib.sha256(signature).hexdigest()[:24]
        client = chromadb.PersistentClient(
            path=str(self.settings.vector_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        collection = client.get_or_create_collection(name=name, embedding_function=None, metadata={"hnsw:space": "cosine"})
        embeddings = OpenAIEmbeddings(
            model=self.settings.embedding_model,
            base_url=self.settings.qwen_base_url,
            api_key=self.settings.qwen_api_key,
            check_embedding_ctx_length=False,
            chunk_size=10,
            request_timeout=self.settings.timeout,
            max_retries=1,
        )
        ids = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
        existing = set(collection.get(include=[])["ids"])
        pending = [(key, text) for key, text in zip(ids, texts) if key not in existing]
        for offset in range(0, len(pending), 10):
            batch = pending[offset:offset + 10]
            batch_texts = [text for _, text in batch]
            vectors = embeddings.embed_documents(batch_texts)
            collection.upsert(ids=[key for key, _ in batch], documents=batch_texts, embeddings=vectors)
            logger.info("分类向量已准备 %s/%s", min(offset + 10, len(pending)), len(pending))
        self._collection, self._embeddings = collection, embeddings
        return collection.count()

    def invoke(self, description: str, config=None) -> list[str]:
        """Only the current business summary is sent to the embedding endpoint."""
        self.prepare()
        vector = self._embeddings.embed_query(description)
        result = self._collection.query(
            query_embeddings=[vector], n_results=min(self.settings.top_k, self._collection.count()),
            include=["documents"],
        )
        return result["documents"][0]
