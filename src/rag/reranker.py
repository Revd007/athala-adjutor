from sentence_transformers import CrossEncoder
from logger import logger

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, documents):
        """Rerank documents based on relevance to query."""
        try:
            if not documents:
                return []

            # Prepare query-document pairs
            pairs = [[query, doc] for doc in documents]
            scores = self.model.predict(pairs)

            # Sort documents by score
            ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            logger.info(f"Reranked {len(documents)} documents for query: {query}")
            return [doc for doc, _ in ranked]
        except Exception as e:
            logger.error(f"Reranking error: {e}")
            raise

if __name__ == "__main__":
    reranker = Reranker()
    query = "What is AI?"
    docs = ["AI is intelligence in machines.", "Machine learning is a subset of AI.", "Python is a language."]
    print(reranker.rerank(query, docs))