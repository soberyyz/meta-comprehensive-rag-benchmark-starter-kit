from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np

class ImageRetriever:
    def __init__(self, model_name='openai/clip-vit-base-patch32'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
    def encode_images(self, images):
        inputs = self.processor(images=images, return_tensors='pt').to(self.device)
        with torch.no_grad():
            return self.model.get_image_features(**inputs).cpu().numpy()

    def search(self, image, image_docs, top_k=10):
        """
        执行端到端图像检索
        :param image: 查询图片(PIL图像对象)
        :param image_docs: 候选图片列表(PIL图像对象集合)
        :param top_k: 返回结果数量
        """
        query_emb = self.encode_images([image])
        doc_embs = self.encode_images(image_docs)
        similarities = np.dot(doc_embs, query_emb.T)
        ranked_indices = np.argsort(-similarities.squeeze())
        return [image_docs[i] for i in ranked_indices[:top_k]]