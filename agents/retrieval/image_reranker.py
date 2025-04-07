from transformers import CLIPProcessor, CLIPModel
import torch

class ImageReranker:
    def __init__(self, model_name='openai/clip-vit-large-patch14'):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def rerank(self, query_image, candidate_images, top_k=10, threshold=0.5):
        # 提取query特征
        query_inputs = self.processor(images=query_image, return_tensors='pt')
        with torch.no_grad():
            query_features = self.model.get_image_features(**query_inputs)

        # 批量提取候选特征
        candidate_inputs = self.processor(images=candidate_images, return_tensors='pt')
        with torch.no_grad():
            candidate_features = self.model.get_image_features(**candidate_inputs)

        # 计算相似度矩阵
        similarities = torch.matmul(candidate_features, query_features.T).squeeze()
        sorted_indices = torch.argsort(-similarities)
        
        # 应用阈值过滤
        filtered_indices = [i for i in sorted_indices if similarities[i] > threshold]
        filtered_indices = filtered_indices[:top_k]
        result_images = [candidate_images[i] for i in filtered_indices]
        return result_images