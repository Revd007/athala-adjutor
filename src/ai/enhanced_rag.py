import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, GPT2LMHeadModel, GPT2Tokenizer, AutoModelForSequenceClassification
from src.utils.database_manager import DatabaseManager
from logger import logger
from config import DATA_DIR
import pandas as pd
import os

class EnhancedRAG:
    def __init__(self, model_path=f"{DATA_DIR}/models/rag_model.pt", index_path=f"{DATA_DIR}/processed/rag_index"):
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.gpt_model = GPT2LMHeadModel.from_pretrained(model_path if os.path.exists(model_path) else "gpt2")
        self.embed_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.embed_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.reranker = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.rerank_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.db = DatabaseManager()
        self.index = faiss.IndexFlatL2(384)
        self.documents = []
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            self.documents = pd.read_parquet(f"{index_path}/documents.parquet")["text"].tolist()
        logger.info("EnhancedRAG initialized with pre-trained models and PostgreSQL")

    def embed_text(self, text):
        try:
            inputs = self.embed_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                embedding = self.embed_model(**inputs).last_hidden_state.mean(dim=1).numpy()
            return embedding
        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
            return None

    def index_documents(self, documents):
        try:
            embeddings = [self.embed_text(doc) for doc in documents]
            embeddings = [e for e in embeddings if e is not None]
            for i, (doc, emb) in enumerate(zip(documents, embeddings)):
                self.db.store_rag_metadata(i, doc, emb)
            embeddings = np.vstack(embeddings)
            self.index.add(embeddings)
            self.documents.extend(documents)
            faiss.write_index(self.index, f"{DATA_DIR}/processed/rag_index/index.faiss")
            pd.DataFrame({"text": self.documents}).to_parquet(f"{DATA_DIR}/processed/rag_index/documents.parquet")
            logger.info(f"Indexed {len(documents)} documents")
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")

    def rerank(self, query, documents):
        try:
            inputs = [self.rerank_tokenizer(query, doc, return_tensors="pt", truncation=True, padding=True) for doc in documents]
            scores = []
            for input in inputs:
                with torch.no_grad():
                    score = self.reranker(**input).logits.item()
                scores.append(score)
            ranked_docs = [doc for _, doc in sorted(zip(scores, documents), reverse=True)]
            logger.info(f"Reranked {len(documents)} documents")
            return ranked_docs[:3]
        except Exception as e:
            logger.error(f"Error reranking documents: {str(e)}")
            return documents

    def retrieve(self, query, k=10):
        try:
            query_embedding = self.embed_text(query)
            if query_embedding is None:
                return []
            distances, indices = self.index.search(query_embedding, k)
            retrieved_docs = []
            for i in indices[0]:
                if i < len(self.documents):
                    text, _ = self.db.fetch_rag_metadata(i)
                    if text:
                        retrieved_docs.append(text)
            ranked_docs = self.rerank(query, retrieved_docs)
            logger.info(f"Retrieved and reranked {len(ranked_docs)} documents for query: {query}")
            return ranked_docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []

    def generate(self, query):
        try:
            retrieved_docs = self.retrieve(query)
            context = "\n".join(retrieved_docs)
            prompt = f"Context:\n{context}\n\nQuery: {query}\nAnswer:"
            inputs = self.gpt_tokenizer(prompt, return_tensors="pt")
            outputs = self.gpt_model.generate(inputs["input_ids"], max_length=200, num_return_sequences=1)
            response = self.gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Generated Enhanced RAG response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error generating Enhanced RAG response: {str(e)}")
            return "Maaf, ada error bro!"