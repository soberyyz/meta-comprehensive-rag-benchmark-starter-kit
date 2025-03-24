# CRAG-MM Dataset Documentation 🚀

**CRAG-MM (Comprehensive RAG Benchmark for Multi-modal, Multi-turn)** is a factual visual question-answering dataset that focuses on real-world images and conversation-based question answering. It is designed to help you evaluate and train **retrieval-augmented generation (RAG)** systems for both **single-turn** and **multi-turn** scenarios.

---

## 1. Dataset Highlights

1. **Images**: A mixture of egocentric images captured from RayBan Meta Smart Glasses and images from publicly available sources.  
2. **Domains**: 13 total, ranging from `shopping` to `food` to `math and science`, aiming to test broad knowledge coverage.  
3. **Query Categories**: Simple recognition, multi-hop reasoning, comparison, aggregation, and more—covering diverse cognitive tasks.  
4. **Turns**: The dataset includes examples with just one Q&A pair (**single-turn**) as well as more extended dialogues with multiple Q&A pairs about the same image (**multi-turn**).  
5. **Image Quality Variations**: e.g. `normal`, `low light`, `blurred`, ensuring robustness to real-world conditions.

---

## 2. Data Splits

CRAG-MM is typically divided into four main splits to support model development cycles:

- **train**
- **validation**
- **test**
- **sample** (small subset for quick debugging or prototyping)

---

## 3. Data Structure

### Single-Turn Format

An example entry (one Q&A about a single image) may look like:

```json
{
  "session_id": "string",
  "image": "path/to/image.jpg",
  "turns": [
    {
      "interaction_id": "string",
      "domain": "string", 
      "query_category": "string",
      "dynamism": "string",
      "query": "string",
      "image_quality": "string"
    }
  ],
  "answers": [
    {
      "interaction_id": "string",
      "ans_full": "string"
    }
  ]
}
```
- **`session_id`**: Unique identifier for this example.  
- **`image`**: The image file (or path) for the single turn.  
- **`turns`**: Contains exactly one turn in single-turn format.  
  - **`interaction_id`** links to the matching answer in `answers`.  
  - **`domain`**, **`query_category`**, **`dynamism`**, **`image_quality`**: Categorical labels describing the question and environment.  
- **`answers`**: The list (usually length 1) containing the ground-truth answer.  

### Multi-Turn Format

An example entry (conversation with multiple Q&As on the same image) may look like:

```json
{
  "session_id": "string",
  "image": "path/to/image.jpg",
  "turns": [
    {
      "interaction_id": "turn0",
      "domain": "string",
      "query_category": "string",
      "dynamism": "string",
      "query": "string",
      "image_quality": "string"
    },
    {
      "interaction_id": "turn1",
      "domain": "string",
      "query_category": "string",
      "dynamism": "string",
      "query": "string",
      "image_quality": "string"
    }
  ],
  "answers": [
    {
      "interaction_id": "turn0",
      "ans_full": "string"
    },
    {
      "interaction_id": "turn1",
      "ans_full": "string"
    }
  ]
}
```
- **`session_id`**: Unique identifier for the entire conversation.  
- **`image`**: One image shared across all turns.  
- **`turns`**: Each user query is listed as a separate object in this array.  
- **`answers`**: Ground-truth answers, matching by `interaction_id`.

---

## 4. Accessing the Dataset

### Option A: Hugging Face Datasets Library

```python
from datasets import load_dataset

# For single-turn data
single_turn_data = load_dataset("crag-mm-2025/crag-mm-single-turn-public", revision="v0.1.0")

# For multi-turn data
multi_turn_data = load_dataset("crag-mm-2025/crag-mm-multi-turn-public", revision="v0.1.0")

# Access a sample
print(single_turn_data["sample"][0])
```

### Option B: Direct Download

The dataset may also be downloadable as an archive (e.g., `.tar.gz`). Once you extract it, you’ll have the same JSON structures as described above.

---

## 5. Usage Tips

1. **Preprocessing**: If your model uses images, consider resizing or normalizing them.  
2. **Handling Multi-turn**: Make sure to maintain conversation context. Each turn depends on previous Q&A pairs in the same `session_id`.  
3. **Evaluation**: Evaluate your responses with either strict exact-match accuracy or semantic metrics like LLM-based scoring.  
4. **Retrieval-Augmentation**: Questions in CRAG-MM can require external knowledge. You can attach a retrieval pipeline (e.g., web search or knowledge-base search) to help answer complex queries.

---

## 6. License & Citation

- **License**: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0).  
- **Citation**: If you use CRAG-MM in your research, please cite:

```bibtex
@inproceedings{crag-mm-2025,
  title = {CRAG-MM: A Comprehensive RAG Benchmark for Multi-modal, Multi-turn Question Answering},
  author = {CRAG-MM Team},
  year = {2025},
  url = {https://www.aicrowd.com/challenges/meta-crag-mm-challenge-2025}
}
```

---

### Happy Exploring!

Whether you are building a next-gen virtual assistant or experimenting with multimodal retrieval, CRAG-MM offers a versatile and challenging environment to test and push the boundaries of your AI models. Good luck!