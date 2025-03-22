import random
import string
from typing import Dict, List, Optional


class DummyAgent:
    def __init__(self, model_id="meta-llama/Llama-3.2-11B-Vision-Instruct", max_gen_len=64):
        """Initialize the agent with a model ID from HF. As per the challenge requirement, we only use LLaMA model"""
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MllamaForConditionalGeneration.from_pretrained(self.model_id, 
                                                                    torch_dtype=torch.bfloat16, 
                                                                    device_map='auto')
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.max_gen_len = max_gen_len
    
    def generate_response(self, query: str, image: Optional[str] = None, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate a response using the model given the query and image. We currently don't consider multi_turn
            Args: 
                Query: The question of the current turn. 
                Image: The image of the conversation session. There is only one image per conversation. 
                Conversation_history: For multi-turn conversation only. Questions and answers of previous turns. 
                    For single-turn, this is []. 
                    For multi-turn, this is a list of two lists. The first contains questions, and the second contains answers. 
        """
        prompt = """You are a helpful assistant that answers user questions. 
        Please answer the following question given the image. 
        """
        messages = [{"role": "user", "content": [
                {"type": "image"},
            ]}
        ]
        if conversation_history != []:
            # add conversation history to the 'messages'
            turn_questions, turn_answers = conversation_history
            for q, a in zip(turn_questions, turn_answers):
                messages.append(
                    {"role": "user", "content": [
                        {"type": "text", "text": q['query']}
                    ]}
                )
                messages.append(
                    {"role": "assistant", "content": [
                        {"type": "text", "text": a['ans_full']}
                    ]}
                ) 
        messages.append(
            {"role": "user", "content": [
                {"type": "text", "text": prompt + query}
            ]}
        )
        # print('QUERY', query)
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        # print('input_text', input_text)
        # print('conversation_history', conversation_history)
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=self.max_gen_len)
        answer = self.processor.decode(output[0])
        return answer.split('<|start_header_id|>assistant<|end_header_id|>')[-1].split('<|eot_id|>')[0]

