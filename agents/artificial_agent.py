from typing import Dict, List, Any, Tuple
from PIL import Image
from agents.base_agent import BaseAgent
from cragmm_search.search import UnifiedSearchPipeline
from crag_web_result_fetcher import WebSearchResult
import vllm
import os
import torch

AICROWD_SUBMISSION_BATCH_SIZE = 2
# GPU utilization settings 
# Change VLLM_TENSOR_PARALLEL_SIZE during local runs based on your available GPUs
# For example, if you have 2 GPUs on the server, set VLLM_TENSOR_PARALLEL_SIZE=2. 
# You may need to uncomment the following line to perform local evaluation with VLLM_TENSOR_PARALLEL_SIZE>1. 
# os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

#### Please ensure that when you submit, VLLM_TENSOR_PARALLEL_SIZE=1. 
VLLM_TENSOR_PARALLEL_SIZE = 1 
VLLM_GPU_MEMORY_UTILIZATION = 0.7 


# These are model specific parameters to get the model to run on a single NVIDIA L40s GPU
MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 2
MAX_GENERATION_TOKENS = 75

# Number of search results to retrieve
NUM_SEARCH_RESULTS = 3
class IntelligentAgent(BaseAgent):
    """
   
    """
    def __init__(
            self, 
            search_pipeline: UnifiedSearchPipeline, 
            model_name="meta-llama/Llama-3.2-11B-Vision-Instruct", 
            max_gen_len=64):
        """
        Initialize the IntelligentAgent.

        """
        super().__init__(search_pipeline)
        self.model_name = model_name
        self.max_gen_len = max_gen_len
        self.initialize_models()

    def initialize_models(self):
        # Initialize the model with vLLM
        self.llm = vllm.LLM(
            self.model_name,
            tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE, 
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION, 
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=MAX_NUM_SEQS,
            trust_remote_code=True,
            dtype="bfloat16",
            enforce_eager=True,
            limit_mm_per_prompt={
                "image": 1 
            } # In the CRAG-MM dataset, every conversation has at most 1 image
        )
        self.tokenizer = self.llm.get_tokenizer()
        
        print("Loaded models")

    def should_search_and_answer(
        self,
        queries: List[str],
        images: List[Image.Image]
    ) -> List[Tuple[bool, bool]]:
        """
        Determines whether each query-image pair needs retrieval and if the model can answer without search.

        Returns:
            List[Tuple[bool, bool]]: List of (can_answer, needs_search) flags for each input.
        """
        prompt_template = (
            "Given the image and the question, answer with two words:\n"
            "'Yes, Yes' if the image contains enough information to answer and no search is needed.\n"
            "'Yes, No' if the image contains enough info and no search is needed.\n"
            "'No, Yes' if the image does not contain enough info and needs search.\n"
            "'No, No' if the question can't be answered at all.\n"
            "Respond only with two words, separated by a comma."
        )

        prompts = []
        for query, image in zip(queries, images):
            messages = [
                {"role": "system", "content": prompt_template},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]},
            ]
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            prompts.append({
                "prompt": prompt,
                "multi_modal_data": {"image": image}
            })

        outputs = self.llm.generate(
            prompts,
            sampling_params=vllm.SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=5,
                skip_special_tokens=True
            )
        )

        results = []
        for output in outputs:
            response = output.outputs[0].text.strip().lower()
            can_answer = "yes" in response.split(",")[0]
            needs_search = "yes" in response.split(",")[1]
            results.append((can_answer, needs_search))
        return results
    
    def get_batch_size(self) -> int:
        """
        Determines the batch size used by the evaluator when calling batch_generate_response.
        
        """
        return AICROWD_SUBMISSION_BATCH_SIZE
    
    def batch_summarize_images(self, images: List[Image.Image]) -> List[str]:
        """
        Generate brief summaries for a batch of images to use as search keywords.
        
        This method efficiently processes all images in a single batch call to the model,
        resulting in better performance compared to sequential processing.
        
        Args:
            images (List[Image.Image]): List of images to summarize.
            
        Returns:
            List[str]: List of brief text summaries, one per image.
        """
        # Prepare image summarization prompts in batch
        summarize_prompt = "Please summarize the image with one sentence that describes its key elements."
        
        inputs = []
        for image in images:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that accurately describes images. Your responses are subsequently used to perform a web search to retrieve the relevant information about the image."},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": summarize_prompt}]},
            ]
            
            # Format prompt using the tokenizer
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            
            inputs.append({
                "prompt": formatted_prompt,
                "multi_modal_data": {
                    "image": image
                }
            })
        
        # Generate summaries in a single batch call
        outputs = self.llm.generate(
            inputs,
            sampling_params=vllm.SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=30,  # Short summary only
                skip_special_tokens=True
            )
        )
        
        # Extract and clean summaries
        summaries = [output.outputs[0].text.strip() for output in outputs]
        print(f"Generated {len(summaries)} image summaries")
        return summaries
    
    def  prepare_rag_enhanced_inputs(
        self, 
        queries: List[str], 
        images: List[Image.Image], 
        image_summaries: List[str],
        message_histories: List[List[Dict[str, Any]]]
    ) -> List[dict]:
        """
        Prepare RAG-enhanced inputs for the model by retrieving relevant information in batch.
        
        This method:
        1. Uses image summaries combined with queries to perform effective searches
        2. Retrieves contextual information from the search_pipeline
        3. Formats prompts incorporating this retrieved information
        
        Args:
            queries (List[str]): List of user questions.
            images (List[Image.Image]): List of images to analyze.
            image_summaries (List[str]): List of image summaries for search.
            message_histories (List[List[Dict[str, Any]]]): List of conversation histories.
            
        Returns:
            List[dict]: List of input dictionaries ready for the model.
        """
        # Batch process search queries
        search_results_batch = []
        
        # Create combined search queries for each image+query pair
        search_queries = [f"{query} {summary}" for query, summary in zip(queries, image_summaries)]
        
        # Retrieve relevant information for each query
        for i, search_query in enumerate(search_queries):
            results = self.search_pipeline(search_query, k=NUM_SEARCH_RESULTS)
            search_results_batch.append(results)
        
        # Prepare formatted inputs with RAG context for each query
        inputs = []
        for idx, (query, image, message_history, search_results) in enumerate(
            zip(queries, images, message_histories, search_results_batch)
        ):
            # Create system prompt with RAG guidelines
            SYSTEM_PROMPT = ("You are a helpful assistant that truthfully answers user questions about the provided image."
                           "Keep your response concise and to the point. If you don't know the answer, respond with 'I don't know'.")
            
            # Add retrieved context if available
            rag_context = ""
            if search_results:
                rag_context = "Here is some additional information that may help you answer:\n\n"
                for i, result in enumerate(search_results):
                    # WebSearchResult is a helper class to get the full page content of a web search result.
                    #
                    # It first checks if the page content is already available in the cache. If not, it fetches  
                    # the full page content and caches it.
                    #
                    # WebSearchResult adds `page_content` attribute to the result dictionary where the page 
                    # content is stored. You can use it like a regular dictionary to fetch other attributes.
                    #
                    # result["page_content"] for complete page content, this is available only via WebSearchResult
                    # result["page_url"] for page URL
                    # result["page_name"] for page title
                    # result["page_snippet"] for page snippet
                    # result["score"] relavancy with the search query
                    result = WebSearchResult(result)
                    snippet = result.get('page_snippet', '')
                    if snippet:
                        rag_context += f"[Info {i+1}] {snippet}\n\n"
                
            # Structure messages with image and RAG context
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "image"}]}
            ]
            
            # Add conversation history for multi-turn conversations
            if message_history:
                messages = messages + message_history
                
            # Add RAG context as a separate user message if available
            if rag_context:
                messages.append({"role": "user", "content": rag_context})
                
            # Add the current query
            messages.append({"role": "user", "content": query})
            
            # Apply chat template
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            
            inputs.append({
                "prompt": formatted_prompt,
                "multi_modal_data": {
                    "image": image
                }
            })
        
        return inputs

    def batch_generate_response(
        self,
        queries: List[str],
        images: List[Image.Image],
        message_histories: List[List[Dict[str, Any]]],
    ) -> List[str]:
        print(f"Processing batch of {len(queries)} queries with smart RAG")

        decision_flags = self.should_search_and_answer(queries, images)
        print("Decision flags:", decision_flags)

        responses = []

        # Step 1: Use batch summarize only for those needing search
        needs_search_flags = [needs_search for _, needs_search in decision_flags]
        images_to_summarize = [img for img, flag in zip(images, needs_search_flags) if flag]
        summaries = self.batch_summarize_images(images_to_summarize) if images_to_summarize else []

        # Map summaries back to full query list
        summary_iter = iter(summaries)
        full_summaries = [
            next(summary_iter) if flag else "" for flag in needs_search_flags
        ]

        # Step 2: For each item, decide how to build the input
        for i, (query, image, history, (can_answer, needs_search), summary) in enumerate(
            zip(queries, images, message_histories, decision_flags, full_summaries)
        ):
            if can_answer and not needs_search:
                # Direct response without retrieval
                messages = [{"role": "system", "content": "You are a helpful assistant."}]
                if history:
                    messages.extend(history)
                messages.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]})

                prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                inputs = [{"prompt": prompt, "multi_modal_data": {"image": image}}]
                output = self.llm.generate(inputs, sampling_params=vllm.SamplingParams(
                    temperature=0.1, top_p=0.9, max_tokens=MAX_GENERATION_TOKENS, skip_special_tokens=True))
                responses.append(output[0].outputs[0].text.strip())

            elif needs_search:
                # Full RAG pipeline
                rag_inputs = self.prepare_rag_enhanced_inputs([query], [image], [summary], [history])
                output = self.llm.generate(rag_inputs, sampling_params=vllm.SamplingParams(
                    temperature=0.1, top_p=0.9, max_tokens=MAX_GENERATION_TOKENS, skip_special_tokens=True))
                responses.append(output[0].outputs[0].text.strip())

            else:
                responses.append("I don't know.")

        return responses
    