# Baseline Implementations 🚀

This document highlights the **baseline agents** available in the **`agents`** directory of this repository. These agents showcase different approaches to solving the **Meta CRAG-MM** challenge. We hope you’ll find them a **fun** and **instructive** starting point for your own creations! 🤖

---

## 🎯 Why Baselines?

Baselines demonstrate how to implement agents that comply with the [submission guidelines](../docs/submission.md) and the [agent interface](../agents/README.md). They also provide reference points for **local evaluation** using `local_evaluation.py`. You can experiment with them, adapt their logic, or build entirely new solutions!

---

## 1. **RandomAgent** 🎲

**File**: [`agents/random_agent.py`](../agents/random_agent.py)  
**Class Name**: `RandomAgent`

### Key Features

- **Random String Generator**: Produces a random string of letters of variable length (between 2 and 16 characters).  
- **Quick & Simple**: Illustrates the agent interface with minimal overhead.  
- **No Real Intelligence**: Strictly for testing the evaluation pipeline.

### Usage Example

```python
from agents.random_agent import RandomAgent

agent = RandomAgent()
response = agent.generate_response(query="What is this?", image="path/to/image.jpg")
print(response)  # Outputs a random string, e.g. "abDUq hf"
```

> **Fun Fact**: This agent is ideal for verifying that your evaluation setup can handle any agent object without worrying about actual performance! 🎉

---

## 2. **LlamaVisionModel** 🦙

**File**: [`agents/vlm_agent.py`](../agents/vlm_agent.py)  
**Class Name**: `LlamaVisionModel`

### Key Features

- **Vision-Language Model**: Uses `meta-llama/Llama-3.2-11B-Vision-Instruct` via **Hugging Face**.
- **Image + Text**: Processes an image (passed as a file path) plus a user query.
- **Conversation History**: Demonstrates basic handling of multi-turn context.
- **Inference**: Uses `MllamaForConditionalGeneration` and an `AutoProcessor` to generate responses.

### Usage Example

```python
from agents.vlm_agent import LlamaVisionModel

agent = LlamaVisionModel()
response = agent.generate_response(
    query="Where was this photo taken?",
    image="path/to/image.jpg"
)
print(response)
```

> **Note**: This agent is a straightforward “vision + text” pipeline. For details on how to incorporate your own vision-language models, see [agents/README.md](../agents/README.md).

---

## 3. **SimpleRAGAgent** 🔎

**File**: [`agents/rag_agent.py`](../agents/rag_agent.py)  
**Class Name**: `SimpleRAGAgent`

### Key Features

- **Retrieval-Augmented Generation**: Uses `UnifiedSearchPipeline` to gather external text snippets from both web pages and images (via a search index).
- **Automatic Summaries**: Summarizes the image content before searching, then appends search snippets to the final query.
- **Multi-Source**: Intended to show how RAG can integrate **image-based** and **web-based** retrieval in a single agent.

### Usage Example

```python
from agents.rag_agent import SimpleRAGAgent

agent = SimpleRAGAgent()
response = agent.generate_response(
    query="What type of car is this, and how can I fix its broken tail light?",
    image="path/to/car_photo.jpg"
)
print(response)
```

> **Pro Tip**: Because RAG methods involve external data, you may want to refine your prompting and ranking to ensure relevant search hits are included. See the code for further insights! 🤓

---

## 🧰 Additional Resources

1. **[Submission Guidelines](../docs/submission.md)** – Explains how to structure your repo and push your agent for evaluation.
2. **[Agent Development Guide](../agents/README.md)** – Details how to create or modify an agent, including the `BaseAgent` interface.
3. **[Local Evaluation Script](../local_evaluation.py)** – Lets you test agents on the CRAG-MM dataset splits (e.g., `sample`) for quick iterations.

---

## 🔧 How to Switch Between Baselines

1. Open [`agents/user_config.py`](../agents/user_config.py).
2. Update the import and assignment:
   ```python
   from agents.rag_agent import SimpleRAGAgent
   UserAgent = SimpleRAGAgent
   ```
3. Run local evaluation:
   ```bash
   python local_evaluation.py --dataset_type single-turn --split sample --num_eval=10
   ```

---

## 🚀 Ready to Build Your Own?

1. **Pick a Baseline** that resonates with your approach.  
2. **Clone** it or **start fresh** with the [`BaseAgent`](../agents/base_agent.py).  
3. **Implement** your own fancy method—be it advanced prompt engineering, improved retrieval logic, or cutting-edge generative LLM.  
4. **Submit** your creation by following the instructions in [submission.md](../docs/submission.md)!  

---

**Enjoy hacking on these baselines, and may your answers be ever grounded in truth!** 🏆
