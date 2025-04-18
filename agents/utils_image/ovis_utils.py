import torch
from PIL import Image
from modelscope import AutoModelForCausalLM

class OvisProcessor:
    def __init__(self, model_path="/root/model_weight/Ovis2-1B"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            multimodal_max_length=32768,
            trust_remote_code=True
        ).cuda()
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()

    def _process_query(self, image, prompt):
        """统一处理图像和prompt的通用方法"""
        # 构建输入参数
        images = [image]
        query = f'<image>\n{prompt}'
        
        # 预处理输入
        _, input_ids, pixel_values = self.model.preprocess_inputs(
            query, 
            images,
            max_partition=9  # 单图像默认分区数
        )
        
        # 准备attention mask
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        
        # 设备转换
        input_ids = input_ids.unsqueeze(0).to(self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.model.device)
        
        if pixel_values is not None:
            pixel_values = pixel_values.to(
                dtype=self.visual_tokenizer.dtype,
                device=self.visual_tokenizer.device
            )
        
        # 生成参数配置
        gen_config = {
            "max_new_tokens": 1024,
            "do_sample": False,
            "top_p": None,
            "top_k": None,
            "temperature": None,
            "repetition_penalty": None,
            "eos_token_id": self.model.generation_config.eos_token_id,
            "pad_token_id": self.text_tokenizer.pad_token_id,
            "use_cache": True
        }

        # 执行推理
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                pixel_values=[pixel_values],
                attention_mask=attention_mask,
                **gen_config
            )[0]
            
        return self.text_tokenizer.decode(output_ids, skip_special_tokens=True)

    def get_image_content(self, image: Image.Image, user_query: str) -> str:
        """生成图像的详细描述"""
        return self._process_query(
            image=image,
            prompt="Describe in detail about \"{}\".No more than 10 words.".format(user_query)
        )

    def get_image_text(self, image: Image.Image) -> str:
        """提取图像中的所有文本"""
        return self._process_query(
            image=image,
            prompt="Accurately extract all text from the given image. If no text is present, respond with an empty string. Do not include any additional information or formatting in the response."
        )

# 使用示例
if __name__ == "__main__":
    processor = OvisProcessor()
    
    # 处理单张图像
    image = Image.open("/root/b784194770c684d142b5297ebf7874c0.jpg")
    
    # 获取图像描述
    import time
    start_time = time.time()
    description = processor.get_image_content(image, "who in picture")
    single_time = time.time()
    print("Image Description:", description)
    
    # 提取图像文本
    extracted_text = processor.get_image_text(image)
    end_time = time.time()
    print("Extracted Text:", extracted_text)

    print("耗时：", single_time - start_time, end_time - single_time)