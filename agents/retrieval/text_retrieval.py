from sentence_transformers import SentenceTransformer
from loguru import logger
import numpy as np
import torch
import faiss

class TextRetriever:
    def __init__(self, model_name='BAAI/bge-base-en-v1.5'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name).to(self.device)
        self.faiss_index = None
        self.id_to_doc = {}
        logger.info(f'Loaded model on {self.device.upper()}')

    def build_lib(self, documents):
        """
        构建文档库的FAISS索引
        :param documents: 文档对象列表，需包含content属性
        """
        doc_contents = [doc.content for doc in documents]
        embeddings = self.model.encode(doc_contents, normalize_embeddings=True)

        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        
        self.faiss_index.add(embeddings.astype('float32'))
        self.id_to_doc = {i: doc for i, doc in enumerate(documents)}
        logger.success(f'成功构建 {len(documents)} 条文档的FAISS索引')

    def search_from_lib(self, query_text, top_k=10):
        """
        从预构建的文档库中进行检索
        :param query_text: 查询文本
        :param top_k: 返回结果数量
        """
        if self.faiss_index is None:
            raise ValueError("请先调用build_lib方法构建文档库索引")
        
        query_emb = self.encode_queries([query_text])
        distances, indices = self.faiss_index.search(query_emb.astype('float32'), top_k)
        
        return [self.id_to_doc[i] for i in indices[0] if i in self.id_to_doc]

    def encode_queries(self, queries):
        return self.model.encode(queries, normalize_embeddings=True, convert_to_tensor=True).to(self.device).cpu().numpy()

    def search(self, query_text, documents, top_k=10):
        """
        执行端到端文本检索
        :param query_text: 原始查询文本字符串
        :param documents: 待检索文档对象列表
        :param top_k: 返回结果数量
        """
        query_emb = self.encode_queries([query_text])
        doc_embeddings = self.model.encode([doc.content for doc in documents])
        similarities = np.dot(doc_embeddings, query_emb.T)
        ranked_indices = np.argsort(-similarities.squeeze())
        return [documents[i] for i in ranked_indices[:top_k]]