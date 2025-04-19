import torch
import requests
from PIL import Image
from typing import Dict, List, Optional
from transformers import MllamaForConditionalGeneration, AutoProcessor
from typing import Optional, Dict, List, Any, Callable
from cragmm_search.search import UnifiedSearchPipeline

from agents.base_agent import BaseAgent
class testAgent(BaseAgent):
    def __init__(
        self, model_id="meta-llama/Llama-3.2-11B-Vision-Instruct", max_gen_len=64
    ):
        """Initialize the agent with a model ID from HF. As per the challenge requirement, we only use LLaMA model"""
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.max_gen_len = max_gen_len
        self.search_pipeline = UnifiedSearchPipeline(
            text_model_name="sentence-transformers/all-MiniLM-L6-v2",
            image_model_name="openai/clip-vit-base-patch16",
            web_hf_dataset_id="crag-mm-2025/web-search-index-public",
            image_hf_dataset_id="crag-mm-2025/image-search-index-public",
        )

    def generate_response(
        self,
        query: str,
        image: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        summary_image_prompt = """
        You are an expert in the joint understanding of images and text. The user will provide a picture, and you are requested to conduct an in-depth analysis from multiple angles, including but not limited to the following:
        1. The main objects, characters, animals, scenes and other visual elements visible in the picture;
        2. The text content in the picture (including titles, labels, slogans, logos, subtitles, handwriting, etc.), please fully extract and explain the meaning;
        3. The relationship and linkage between vision and text (such as whether the text explains the image, whether there is irony, supplement, contrast, etc.);
        4. The style characteristics of the picture (such as cartoons, realism, commercials, posters, screenshots, etc.);
        5. The information, emotions, opinions or story background that the image may convey;
        6. If any, analyze the potential uses behind the image (such as social media content, commercial propaganda, teaching materials, etc.).
        Please output as detailed as possible to ensure that no meaningful details in the image are missed. 
        Please summarize the image with one sentence. 
        Please output the summary only.
        """
        summarize_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": summary_image_prompt}]},
        ]
        summarize_input_text = self.processor.apply_chat_template(
            summarize_messages, add_generation_prompt=True
        )
        summarize_inputs = self.processor(
            image, summarize_input_text, add_special_tokens=False, return_tensors="pt"
        ).to(self.model.device)
        summarize_output = self.model.generate(
            **summarize_inputs, max_new_tokens=self.max_gen_len
        )
        summarize_answer = self.processor.decode(summarize_output[0])
        summarize_answer = summarize_answer.split(
            "<|start_header_id|>assistant<|end_header_id|>"
        )[-1].split("<|eot_id|>")[0]
        prompt = """You are a helpful assistant that answers user questions. 
        Please answer the following question given the image. 
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                ],
            }
        ]
        if conversation_history != []:
            # add conversation history to the 'messages'
            turn_questions, turn_answers = conversation_history
            for q, a in zip(turn_questions, turn_answers):
                messages.append(
                    {"role": "user", "content": [{"type": "text", "text": q["query"]}]}
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": a["agent_response"]}],
                    }
                )
        messages.append(
            {"role": "user", "content": [{"type": "text", "text": prompt + query}]}
        )
        rag_prompt = (
            "Here are some other relevant images and information that may help you.\n\n"
        )

        # According to official community answer, LLaMA-VL does not work well with multiple images in the same conversation,
        # Ref: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct/discussions/43#66f98f742094ed9e5f5107d4
        # so we won't put multiple images in the same conversation.

        # call the web search api for relevant texts.
        search_results = self.search_pipeline(summarize_answer, k=3)

        for result in search_results:
            rag_prompt += result["page_snippet"] + "\n\n"
            print("-------------------------------------------------------")
            print(result["page_snippet"])
        messages.append(
            {"role": "user", "content": [{"type": "text", "text": rag_prompt}]}
        )
        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            image, input_text, add_special_tokens=False, return_tensors="pt"
        ).to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=self.max_gen_len)
        answer = self.processor.decode(output[0])
        return answer.split("<|start_header_id|>assistant<|end_header_id|>")[-1].split(
            "<|eot_id|>"
        )[0]


