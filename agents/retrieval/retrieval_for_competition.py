from typing import List, Union
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from .text_retrieval import TextRetriever
from .image_retrieval import ImageRetriever
from .text_reranker import TextReranker
from .image_reranker import ImageReranker
from .cross_modal_retrieval import CrossModalRetriever

class CompetitionRetriever:
    def __init__(self, 
                 text_model_name: str = 'BAAI/bge-base-en-v1.5',
                 image_model_name: str = 'clip-ViT-B-32',
                 text_reranker_name: str = 'BAAI/bge-reranker-base',
                 image_reranker_name: str = 'clip-ViT-B-32'):
        
        self.text_retriever = TextRetriever(text_model_name)
        self.image_retriever = ImageRetriever(image_model_name)
        self.text_reranker = TextReranker(text_reranker_name)
        self.image_reranker = ImageReranker(image_reranker_name)
        self.cross_modal_retriever = CrossModalRetriever()
    
    def merge_recall_results(self, results: List[List[Union[str, object]]]) -> List[Union[str, object]]:
        merged = {}
        for query_results in results:
            for item in query_results:
                key = item.content if hasattr(item, 'content') else str(item)
                if key not in merged:
                    merged[key] = item
        return list(merged.values())

    def multi_modal_search(self,
                          queries, 
                          candidates,
                          top_k_recall=50,
                          top_k_rerank=10,
                          rerank_threshold=0.5,
                          use_modal_search=False):
        
        # 多模态召回
        recall_results = []
        for query in queries:
            if isinstance(query, str):
                if type(candidates[0]) != str:
                    raise ValueError("文本搜索的情况下，候选文本必须为字符串格式")
                results = self.text_retriever.search(query, candidates, top_k=top_k_recall)
            else:
                if use_modal_search:
                    if type(candidates[0]) != str:
                        raise ValueError("使用跨模态检索时，候选文本必须为字符串格式")
                    results = self.image_search_query(query, candidates, top_k=top_k_recall)
                else:
                    if type(candidates[0]) == str:
                        raise ValueError("图片搜索的情况下，候选文本必须为图片格式")
                    results = self.image_retriever.search(query, candidates, top_k=top_k_recall)
            recall_results.extend(results)
        
        # 结果去重
        unique_results = self.merge_recall_results([recall_results])

        # 使用首个query进行精排
        if len(queries) == 0:
            raise ValueError("至少需要提供一个查询")
        
        main_query = queries[0]
        if isinstance(main_query, str):
            return self.text_reranker.rerank(
                main_query, 
                unique_results, 
                top_k=top_k_rerank,
                threshold=rerank_threshold
            )
        else:
            return self.image_reranker.rerank(
                main_query, 
                unique_results, 
                top_k=top_k_rerank,
                threshold=rerank_threshold
            )

    def image_search_query(self,
                          query_image,
                          candidate_queries,
                          top_k):
        """
        图像到文本的跨模态检索
        :param query_image: 输入图像(Numpy数组格式)
        :param candidate_queries: 候选文本查询列表
        :param top_k: 返回结果数量
        """
        return self.cross_modal_retriever.search(
            query_image=query_image,
            candidate_queries=candidate_queries,
            top_k=top_k
        )

    def build_unified_index(self, documents: List[object]):
        """构建统一的多模态索引"""
        # （需要根据具体存储格式补充实现）
        raise NotImplementedError("统一索引构建功能待实现")