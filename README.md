# Synopsis Checker

## 1. Synopsis Checker

**Synopsis Checker** is a privacy-focused Gen AI powered application designed to quickly evaluate the quality of a synopsis based on four key properties: **Content Coverage**, **Clarity**, **Coherence**, and **Factual Consistency**.  
It supports both **local LLM inference using GGUF models** and **OpenAI ChatGPT API** — simply toggle between the two within the UI.

---

## 2. Setup

To run this app locally on your system, follow these steps:

### Clone and Setup Virtual Environment

#### Windows
```bash
git clone https://github.com/shubhendam/Synopsis_Checker.git
cd Synopsis_Checker
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux
```bash
git clone https://github.com/shubhendam/Synopsis_Checker.git
cd Synopsis_Checker
python3 -m venv venv
source venv/bin/activate
```

#### Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 📥 Download LLM & Embedding Models

- [LLaMA 3.2 3B (GGUF)](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf?download=true)  
- [Nomic Embedding v1.5 (GGUF)](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.f16.gguf?download=true)

Place both `.gguf` files in the `models/` directory.


## 3. Demo

Run the app with:

```bash
streamlit run app.py
```
### In the UI: 

1. Select LLM mode: **Local** or **ChatGPT**
2. Upload an article (`.txt` or `.pdf`) and its synopsis (`.txt`)
3. Click **Analyze** to generate scores and feedback. 

![Demo in Action][(assets/demo.gif](https://www.youtube.com/watch?v=VJfqVIYfrfc))


## 4. Scoring Methodology & Privacy Protection Strategy

### 4.1 Scoring Methodology

**We generate an intelligent semantic summary of the article before sending it to the LLM for evaluation:**

1. We first break the article into paragraphs, and then into individual sentences:
   ```python
   sentences = [sent.text.strip() for sent in nlp(para).sents if sent.text.strip()]
   ```
2. We convert these sentences into vector embeddings and compute an average embedding to capture the paragraph's overall meaning.
   ```python
   avg_embedding = [sum(col) / len(col) for col in zip(*sentence_embeddings)]
   ```
3. We then compute cosine similarity between each sentence and the paragraph’s average to find the most meaningful sentences.
4. The top sentences are dynamically selected based on paragraph size to create a compressed summary rich in facts and structure.
5. This summary (instead of the full article) is sent to the LLM alongside the user-uploaded synopsis to helps reduce token consumption and avoids overwhelming the model with irrelevant data, ensuring focused, efficient evaluations.

### 4.2 Privacy Protection Strategy
**When using OpenAI or other third-party APIs, we apply anonymization to the summaries before sending:**
1. We use spaCy's English NER model to detect sensitive entities.
2. The following types are replaced: {"PERSON", "ORG", "GPE", "DATE"}
3. Replacements are consistent using numbered placeholders like PERSON_X1, ORG_X2, etc.
   ```python
   if ent_type in NER_TYPES_TO_REPLACE:
    key = f"{ent_type}::{token.text}"
    if key not in replacement_map:
        entity_counters[ent_type] += 1
        replacement_map[key] = f"{ent_type}_X{entity_counters[ent_type]}"
   ```
   **Example output**
   ```text
   Original: Elon Musk visited India in January 2024.
   Anonymized: PERSON_X1 visited GPE_X1 in DATE_X1.
   ```
   This ensures that no identifiable data ever leaves the local machine when calling external LLM APIs.


## 5. Rating Parameters
Each synopsis is evaluated based on:
1. **Content Coverage**: How well it captures the main ideas and details from the article.
2. **Clarity**: The readability and structure of the synopsis.
3. **Coherence**: Logical flow and connection between sentences and sections.
4. **Factual Consistency**: Whether the facts mentioned are accurate and consistent with the source article.


## 6. Future Enhancements
1. Support for .docx files in addition to .txt and .pdf
2. Add AI hallucination checker for factual consistency
