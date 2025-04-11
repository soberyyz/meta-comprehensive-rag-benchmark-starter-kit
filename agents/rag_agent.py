from typing import Dict, List, Optional

import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from typing import Optional, Dict, List, Any, Callable
from cragmm_search.search import UnifiedSearchPipeline

from agents.base_agent import BaseAgent


class SimpleRAGAgent(BaseAgent):
    """This class demonstrates the sample use of RAG API for the challenge"""

    """It simply searches the image and the query, and append the retrieved text & image to the query"""

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
            image_model_name="openai/clip-vit-large-patch14-336",
            web_hf_dataset_id="crag-mm-2025/web-search-index-validation",
            image_hf_dataset_id="crag-mm-2025/image-search-index-validation",
        )
        # Please don't change the image_model_name and text_model_name, as the indices are constructed with exactly these models. 

    def generate_response(
        self,
        query: str,
        image: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Generate a response using the model given the query and image. 
        Args:
            Query: The question of the current turn.
            Image: The image of the conversation session. There is only one image per conversation.
            Conversation_history: For multi-turn conversation only. Questions and answers of previous turns.
                For single-turn, this is [].
                For multi-turn, this is a list of two lists. The first contains questions, and the second contains answers.
        """

        # First call the LLM to generate some keywords for the image (for future RAG).
        summarize_prompt = "Please summarize the image with one sentence. Please output the summary only. "
        summarize_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": summarize_prompt}]},
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
