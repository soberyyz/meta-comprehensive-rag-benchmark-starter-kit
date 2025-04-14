import argparse
import os
from copy import deepcopy
from typing import Any
import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor
from torchvision.transforms.functional import resize


class SamResize:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        h, w, _ = image.shape
        long_side = max(h, w)
        if long_side != self.size:
            return self.apply_image(image)
        else:
            return image.permute(2, 0, 1)

    def apply_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects a torch tensor with shape hxwx3 in float format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.size)
        return resize(image.permute(2, 0, 1), target_size)

    @staticmethod
    def get_preprocess_shape(
        oldh: int, oldw: int, long_side_length: int
    ) -> tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        newh = int(newh + 0.5)
        neww = int(neww + 0.5)
        return (newh, neww)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size})"


class SamEncoder:
    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        opt = ort.SessionOptions()
        if device == "cuda":
            provider = ["CUDAExecutionProvider"]
        elif device == "cpu":
            provider = ["CPUExecutionProvider"]
        else:
            raise ValueError("Invalid device, please use 'cuda' or 'cpu' device.")
        print(f"loading encoder model from {model_path}...")
        self.session = ort.InferenceSession(model_path, opt, providers=provider, **kwargs)
        self.input_name = self.session.get_inputs()[0].name

    def _extract_feature(self, tensor: np.ndarray) -> np.ndarray:
        feature = self.session.run(None, {self.input_name: tensor})[0]
        return feature

    def __call__(self, img: np.array, *args: Any, **kwds: Any) -> Any:
        return self._extract_feature(img)


class SamDecoder:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        target_size: int = 1024,
        mask_threshold: float = 0.0,
        **kwargs
    ):
        opt = ort.SessionOptions()
        if device == "cuda":
            provider = ["CUDAExecutionProvider"]
        elif device == "cpu":
            provider = ["CPUExecutionProvider"]
        else:
            raise ValueError("Invalid device, please use 'cuda' or 'cpu' device.")
        print(f"loading decoder model from {model_path}...")
        self.target_size = target_size
        self.mask_threshold = mask_threshold
        self.session = ort.InferenceSession(model_path, opt, providers=provider, **kwargs)

    @staticmethod
    def get_preprocess_shape(
        oldh: int, oldw: int, long_side_length: int
    ) -> tuple[int, int]:
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        newh = int(newh + 0.5)
        neww = int(neww + 0.5)
        return (newh, neww)

    def run(
        self,
        img_embeddings: np.ndarray,
        origin_image_size: list | tuple,
        point_coords: list | np.ndarray = None,
        point_labels: list | np.ndarray = None,
        boxes: list | np.ndarray = None,
        return_logits: bool = False,
    ):
        input_size = self.get_preprocess_shape(
            *origin_image_size, long_side_length=self.target_size
        )
        if point_coords is not None and point_labels is None and boxes is None:
            raise ValueError("Unable to segment, please input at least one box or point.")
        if img_embeddings.shape != (1, 256, 64, 64):
            raise ValueError("Got wrong embedding shape!")
        if point_coords is not None:
            point_coords = self.apply_coords(
                point_coords, origin_image_size, input_size
            ).astype(np.float32)
            prompts, labels = point_coords, point_labels
        if boxes is not None:
            boxes = self.apply_boxes(
                boxes, origin_image_size, input_size
            ).astype(np.float32)
            box_labels = np.array([[2, 3] for _ in range(boxes.shape[0])], dtype=np.float32).reshape((-1, 2))
            if point_coords is not None:
                prompts = np.concatenate([prompts, boxes], axis=1)
                labels = np.concatenate([labels, box_labels], axis=1)
            else:
                prompts, labels = boxes, box_labels
        input_dict = {
            "image_embeddings": img_embeddings,
            "point_coords": prompts,
            "point_labels": labels,
        }
        low_res_masks, iou_predictions = self.session.run(None, input_dict)
        masks = mask_postprocessing(low_res_masks, origin_image_size)
        if not return_logits:
            masks = masks > self.mask_threshold
        return masks, iou_predictions, low_res_masks

    def apply_coords(self, coords, original_size, new_size):
        old_h, old_w = original_size
        new_h, new_w = new_size
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes, original_size, new_size):
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size, new_size)
        return boxes


def preprocess(x, img_size):
    pixel_mean = [123.675 / 255, 116.28 / 255, 103.53 / 255]
    pixel_std = [58.395 / 255, 57.12 / 255, 57.375 / 255]
    x = torch.tensor(x)
    resize_transform = SamResize(img_size)
    x = resize_transform(x).float() / 255
    x = transforms.Normalize(mean=pixel_mean, std=pixel_std)(x)
    h, w = x.shape[-2:]
    th, tw = img_size, img_size
    assert th >= h and tw >= w
    x = F.pad(x, (0, tw - w, 0, th - h), value=0).unsqueeze(0).numpy()
    return x


def resize_longest_image_size(
    input_image_size: torch.Tensor, longest_side: int
) -> torch.Tensor:
    input_image_size = input_image_size.to(torch.float32)
    scale = longest_side / torch.max(input_image_size)
    transformed_size = scale * input_image_size
    transformed_size = torch.floor(transformed_size + 0.5).to(torch.int64)
    return transformed_size


def mask_postprocessing(
    masks: torch.Tensor, orig_in_size: torch.Tensor
) -> torch.Tensor:
    img_size = 1024
    masks = torch.tensor(masks)
    orig_in_size = torch.tensor(orig_in_size)
    masks = F.interpolate(
        masks,
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )
    prepadded_size = resize_longest_image_size(orig_in_size, img_size)
    masks = masks[..., : int(prepadded_size[0]), : int(prepadded_size[1])]
    orig_in_size = orig_in_size.to(torch.int64)
    h, w = orig_in_size[0], orig_in_size[1]
    masks = F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)
    return masks


def segment_image_per_point(
    img_path: str,
    out_folder: str,
    encoder_model,
    decoder_model,
    model_type="efficientvit-sam-xlt",
):
    os.makedirs(out_folder, exist_ok=True)
    # 初始化编码器和解码器
    encoder = encoder_model
    decoder = decoder_model
    # 读取图像并获取原始尺寸
    raw_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    origin_image_size = raw_img.shape[:2]
    # 根据模型类型预处理图像
    if model_type in ["efficientvit-sam-l0", "efficientvit-sam-l1", "efficientvit-sam-l2"]:
        img = preprocess(raw_img, img_size=512)
    elif model_type in ["efficientvit-sam-xlt"]:
        img = preprocess(raw_img, img_size=1024)
    else:
        raise NotImplementedError
    # 定义提示信息：点、框等
    H, W, _ = raw_img.shape
    grid_size = 3  # 网格大小
    x_step = W / grid_size
    y_step = H / grid_size
    # 生成点网格
    point_coords_list = []
    for i in range(grid_size):
        for j in range(grid_size):
            x = j * x_step + x_step / 2  # 点位于每个网格单元的中心
            y = i * y_step + y_step / 2
            point_coords_list.append([x, y])
    import time

    start_time = time.time()
    # 提取图像特征
    img_embeddings = encoder(img)
    # 并发处理每个点
    masks = []
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = []
        for point_coords in point_coords_list:
            futures.append(
                executor.submit(
                    process_point,
                    point_coords,
                    img_embeddings,
                    decoder,
                    origin_image_size,
                )
            )
        for future in futures:
            masks.append(future.result())
    end_time = time.time()
    print(f"花费时间: {end_time - start_time}")
    # 合并掩码（根据IoU）
    final_masks = merge_masks_by_iou(masks, iou_threshold=0.5)
    # final_masks = masks
    # 保存最终分割掩码和对应的原图切图
    for i, mask in enumerate(final_masks):
        # 将掩码转换为NumPy数组
        mask = mask.cpu().numpy()
        mask = (mask > decoder.mask_threshold).astype(np.uint8) * 255
        mask = mask[0]  # 去掉多余的维度
        # 应用掩码到原图
        masked_image = apply_mask_to_image(raw_img, mask)
        # 保存掩码和切图
        image_path = os.path.join(out_folder, f"masked_image_{i}.png")
        masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(mask_path, mask)
        cv2.imwrite(image_path, masked_image)
        # print(f"Final mask saved in {mask_path}")
        print(f"Masked image saved in {image_path}")


def apply_mask_to_image(image, mask):
    """
    将掩码应用到原图，背景设置为0。
    参数:
        image (np.ndarray): 原始图像 (H, W, 3)
        mask (np.ndarray): 掩码图像 (H, W), 值为0或255
    返回:
        np.ndarray: 应用掩码后的图像
    """
    # 确保掩码与原图尺寸一致
    assert image.shape[:2] == mask.shape, "Image and mask dimensions do not match!"
    # 将掩码转换为布尔数组
    mask_bool = mask.astype(bool)
    # 创建一个全白的背景图像
    masked_image = np.ones_like(image) * 255
    # 将掩码区域的像素复制到背景图像中
    masked_image[mask_bool] = image[mask_bool]
    return masked_image


def process_point(point_coords, img_embeddings, decoder, origin_image_size):
    """为单个点生成分割掩码"""
    point_coords = np.array([point_coords], dtype=np.float32).reshape(1, 1, 2)  # 形状为 (1, 1, 2)
    point_labels = np.array([[1]], dtype=np.float32).reshape(1, 1)  # 形状为 (1, 1)
    # 使用解码器生成分割掩码
    masks, _, _ = decoder.run(
        img_embeddings=img_embeddings,
        origin_image_size=origin_image_size,
        point_coords=point_coords,
        point_labels=point_labels,
    )
    return masks[0]  # 返回第一个掩码


def calculate_iou(mask1, mask2):
    """计算两个掩码的IoU"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0


def merge_masks_by_iou(masks, iou_threshold=0.5):
    """根据IoU合并掩码"""
    final_masks = []
    for mask in masks:
        merged = False
        for final_mask in final_masks:
            iou = calculate_iou(mask, final_mask)
            if iou > iou_threshold:
                final_mask |= mask  # 合并掩码
                merged = True
                break
        if not merged:
            final_masks.append(mask)
    return final_masks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default="/root/b784194770c684d142b5297ebf7874c0.jpg")
    parser.add_argument("--out_folder", type=str, default="/root/output")
    args = parser.parse_args()
    encoder_model = "/root/onnx/efficientvit_sam_xl1_encoder.onnx"
    decoder_model = "/root/onnx/efficientvit_sam_xl1_decoder.onnx"
    encoder = SamEncoder(model_path=encoder_model)
    decoder = SamDecoder(model_path=decoder_model)
    segment_image_per_point(args.img_path, args.out_folder, encoder, decoder)