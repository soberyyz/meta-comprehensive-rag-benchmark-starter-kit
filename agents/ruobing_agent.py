from typing import Dict, List, Optional

import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from typing import Optional, Dict, List, Any, Callable
from cragmm_search.search import UnifiedSearchPipeline

from agents.base_agent import BaseAgent
from agents.prompts.rag import (
    RAG_BASELINE_PROMPT,
    SYSTEM_PROMPT
)
from agents.prompts.summary import (
    SUMMARY_IMAGE_TEXT,
    SUMMARY_IMAGE_CONTENT
)
from retrieval.retrieval_for_competition import CompetitionRetriever



class TeamAgent(BaseAgent):
    """This class demonstrates the sample use of RAG API for the challenge"""

    """It simply searches the image and the query, and append the retrieved text & image to the query"""

    def __init__(
        self, model_id="meta-llama/Llama-3.2-11B-Vision-Instruct", max_gen_len=256, max_output_words_len=75
    ):
        """Initialize the agent with a model ID from HF. As per the challenge requirement, we only use LLaMA model"""
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.max_gen_len = max_gen_len
        self.max_output_words_len = max_output_words_len
        self.search_pipeline = UnifiedSearchPipeline(
            text_model_name="sentence-transformers/all-MiniLM-L6-v2",
            image_model_name="openai/clip-vit-base-patch16",
            web_hf_dataset_id="crag-mm-2025/web-search-index-public",
            image_hf_dataset_id="crag-mm-2025/image-search-index-public",
        )
        self.competition_retriever = CompetitionRetriever(
            text_model_name="BAAI/bge-base-en-v1.5",
            image_model_name="",
            text_reranker_name="BAAI/bge-reranker-base",
            image_reranker_name="",
        )

    def _get_llm_response(self, query, image=None) -> str:
        if image:
            model_inputs = self.processor(
                image, query, add_special_tokens=False, return_tensors="pt"
            ).to(self.model.device)
        else:
            model_inputs = self.processor(
                query, add_special_tokens=False, return_tensors="pt"
            ).to(self.model.device)
        model_output = self.model.generate(
            **model_inputs, max_new_tokens=self.max_gen_len
        )
        model_answer = self.processor.decode(model_output[0])
        model_answer = model_answer.split(
            "<|start_header_id|>assistant<|end_header_id|>"
        )[-1].split("<|eot_id|>")[0]
        return model_answer

    def _get_image_content(self, image):
        summarize_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": SUMMARY_IMAGE_CONTENT}]},
        ]
        summarize_input_text = self.processor.apply_chat_template(
            summarize_messages, add_generation_prompt=True
        )
        summarize_answer = self._get_llm_response(summarize_input_text, image)
        return summarize_answer
    
    def _get_image_text(self, image):
        summarize_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": SUMMARY_IMAGE_TEXT}]},
        ]
        summarize_input_text = self.processor.apply_chat_template(
            summarize_messages, add_generation_prompt=True
        )
        summarize_answer = self._get_llm_response(summarize_input_text, image)
        return summarize_answer

    def _get_web_search_text(self, web_pages):
        web_search_text = ""
        web_search_text += web_pages["page_snippet"] + "\n\n"
        return web_search_text

    def generate_response(
        self,
        query: str,
        image: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Generate a response using the model given the query and image. We currently don't consider multi_turn
        Args:
            Query: The question of the current turn.
            Image: The image of the conversation session. There is only one image per conversation.
            Conversation_history: For multi-turn conversation only. Questions and answers of previous turns.
                For single-turn, this is [].
                For multi-turn, this is a list of two lists. The first contains questions, and the second contains answers.
        Currently, it works for single-turn primarily because multi-turn data loading has to be changed.
        """

        # First call the LLM to generate some keywords for the image (for future RAG).
        summarize_answer_image_content = self._get_image_content(image)
        summarize_answer_image_text = self._get_image_text(image)
        image_info = "There is a image.\n\n The image content can be concluded as:\n"+summarize_answer_image_content + '\n\n'
        image_info += "The text in the image can be concluded as:\n"+summarize_answer_image_text

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
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
        
        # call the web search api for relevant texts.
        search_results_from_image_info = self.search_pipeline(image_info, k=3)
        search_results_from_user_query = self.search_pipeline(query, k=3)
        web_search_results = []
        for result in search_results_from_image_info + search_results_from_user_query:
            web_search_results.append(self._get_web_search_text(result)) # TODO: 完善一下web处理逻辑@xiaoli

        # call the image search mock API
        search_results_from_image = self.search_pipeline(image, k=3) # TODO: 完善一下image search逻辑@xiaoli
        context_str = "" # TODO: 将所有来源进行集合@keke
        # put them in llm prompt.
        llm_prompt = RAG_BASELINE_PROMPT.format(
            token_limit=self.max_output_words_len, 
            context_str=context_str, 
            image_content=summarize_answer_image_content,
            image_text=summarize_answer_image_text,
            query=query
        )
        messages.append(
            {"role": "user", "content": [{"type": "text", "text": llm_prompt}]}
        )
        # put image in the last.
        messages.append({
                "role": "user",
                "content": [
                    {"type": "image"},
                ],
        })
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
