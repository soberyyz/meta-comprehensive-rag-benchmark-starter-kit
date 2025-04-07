from typing import List, Union
import numpy as np
from loguru import logger
from PIL import Image
from sentence_transformers import SentenceTransformer
import torch

class CrossModalRetriever:
    def __init__(self, model_name: str = 'openai/clip-vit-large-patch14'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=self.device)
        logger.info(f'Loaded CLIP model on {self.device.upper()}')

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """编码图像输入为嵌入向量"""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        return self.model.encode(image, convert_to_tensor=True).cpu().numpy()

    def encode_text(self, text: str) -> np.ndarray:
        """编码文本输入为嵌入向量"""
        return self.model.encode(text, convert_to_tensor=True).cpu().numpy()

    def search(
        self, 
        query_image: np.ndarray,
        candidate_queries: List[str],
        top_k: int = 10,
        threshold: float = 0.25
    ) -> List[Union[str, None]]:
        """
        跨模态检索方法
        :param query_image: 输入图像(Numpy数组格式)
        :param candidate_queries: 候选文本查询列表
        :param top_k: 返回结果数量
        :param threshold: 相似度分数阈值
        """
        # 编码查询图像
        image_emb = self.encode_image(query_image)
        
        # 编码所有候选文本
        text_embs = self.model.encode(candidate_queries, convert_to_tensor=True)
        
        # 计算相似度矩阵
        similarities = torch.matmul(text_embs, torch.tensor(image_emb).to(self.device).T)
        
        # 过滤和排序结果
        filtered_indices = torch.where(similarities >= threshold)[0]
        sorted_indices = filtered_indices[torch.argsort(-similarities[filtered_indices])]
        
        # 返回满足阈值的结果
        return [
            candidate_queries[i] 
            for i in sorted_indices[:top_k] 
            if i < len(candidate_queries)
        ]