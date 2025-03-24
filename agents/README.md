# 🤖 CRAG-MM Agent Development Guide

Welcome to the CRAG-MM agent development playground! This directory is where participants can implement their vision-language models for the challenge. 

We recommend that you put everything you need for the agent to run in this repo for a smoother submission/evaluation process. However, you're free to organize your code as you prefer.

## 🎯 Structure of This Directory

```
agents/
├── base_agent.py      # Base class that all agents must inherit from
├── random_agent.py    # Sample random response agent
├── rag_agent.py       # Sample RAG-based agent
├── vlm_agent.py       # Sample vision-language model agent
└── user_config.py     # Configuration file to specify which agent to use
```

## 🧪 Sample Agents

We've provided three sample agents to help you get started:

1. **RandomAgent** 🎲
   - A simple agent that generates random responses
   - Perfect for testing the evaluation pipeline
   - Located in `random_agent.py`

2. **LlamaVisionModel** 🦙
   - A vision-language model based on Meta's Llama 3.2 11B Vision Instruct model
   - Handles both single-turn and multi-turn conversations
   - Located in `vlm_agent.py`

3. **SimpleRAGAgent** 🔍
   - A RAG (Retrieval-Augmented Generation) based agent
   - Uses unified search pipeline for retrieving relevant information
   - Demonstrates how to combine vision and text search
   - Located in `rag_agent.py`

## 🛠️ Creating Your Own Agent

To create your own agent, follow these steps:

1. Create a new Python file in the `agents` directory
2. Import and inherit from `BaseAgent`:
   ```python
   from agents.base_agent import BaseAgent

   class YourAgent(BaseAgent):
       def __init__(self):
           # Initialize your model here
           # You have 10 minutes for initialization
           pass

       def generate_response(
           self,
           query: str,
           image: Optional[str] = None,
           conversation_history: Optional[List[Dict[str, str]]] = None,
       ) -> str:
           # Implement your response generation logic here
           # You have 10 seconds per response
           return "Your response here"
   ```

### ⚡ Performance Constraints

- **Initialization Time**: 10 minutes maximum
- **Response Time**: 10 seconds per `generate_response` call
- **Memory Usage**: Be mindful of GPU memory if using CUDA

### 📝 Method Signatures

Your agent must implement the following method:

```python
def generate_response(
    self,
    query: str,
    image: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> str:
```

Parameters:
- `query`: The current question from the user
- `image`: Optional image path for vision-language tasks
- `conversation_history`: Optional list of previous Q&A pairs for multi-turn conversations

## 🔧 Configuration

To use your agent:

1. Edit `user_config.py`
2. Import your agent class
3. Assign it to `UserAgent`:
   ```python
   from your_agent import YourAgent
   UserAgent = YourAgent
   ```

## 📦 Model Configuration

During evaluation, internet access will be disabled and `HF_HUB_OFFLINE=1` environment variable will be set. You must specify all required models in `aicrowd.json`:

```json
{
    "challenge_id": "single-source-augmentation",
    "gpu": true,
    "hf_models": [
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "your-org/your-model"
    ]
}
```

### ⚠️ Important Notes

1. **Model Access**: Ensure that the `aicrowd` HuggingFace account has access to all models specified in `hf_models`
2. **Offline Mode**: Any model not specified in `aicrowd.json` will fail during evaluation
3. **Model Download**: We will download all specified models before evaluation starts
4. **Access Control**: If your model is private, make sure to:
   - Grant access to the `aicrowd` HF account
   - Include the model in `hf_models`
   - Use the correct model ID (org/model-name)

## �� Local Evaluation

We provide a `local_evaluation.py` script to evaluate your agent:

```bash
python local_evaluation.py \
    --dataset_type single-turn \
    --split sample \
    --num_eval 100 \
    --show_examples 3
```

Options:
- `--dataset_type`: Choose between "single-turn" or "multi-turn"
- `--split`: Dataset split to use ("train", "validation", "test", "sample")
- `--num_eval`: Number of examples to evaluate
- `--show_examples`: Number of example results to display
- `--disable_llm_judge`: Disable semantic evaluation (uses exact matching only)
- `--eval_model`: OpenAI model for semantic evaluation
- `--output_path`: Save results to a JSON file
- `--num_workers`: Number of parallel evaluation workers

## 🎯 Evaluation Metrics

The evaluation script calculates:
- Exact match accuracy
- Semantic accuracy (using LLM-as-judge)
- Missing rate ("I don't know" responses)
- Hallucination rate
- Overall score

## 🚀 Getting Started

1. Clone the repository
2. Install dependencies
3. Implement your agent
4. Update `user_config.py`
5. Run local evaluation
6. Iterate and improve!

Happy coding! 🚀
