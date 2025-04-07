from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from loguru import logger

class TextReranker:
    def __init__(self, model_name='BAAI/bge-reranker-v2-m3'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.success(f'Loaded model on {self.device.upper()}')

    def rerank(self, query, documents, top_k=10, threshold=0.5):
        pairs = [[query, doc] for doc in documents]
        inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.device)
        with torch.no_grad():
            scores = self.model(**inputs).logits.view(-1).float().tolist()
        filtered = [(doc, score) for doc, score in zip(documents, scores) if score > threshold]
        sorted_results = sorted(filtered, key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]