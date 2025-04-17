# 🧠 Synopsis Checker

## 1. About the App

**Synopsis Checker** is a privacy-focused Gen AI powered application designed to quickly evaluate the quality of a synopsis based on four key properties: **Content Coverage**, **Clarity**, **Coherence**, and **Factual Consistency**.  
It supports both **local LLM inference using GGUF models** and **OpenAI ChatGPT API** — simply toggle between the two within the UI.

---

## 2. Setup

To run this app locally on your system, follow these steps:

### Clone and Setup Virtual Environment

#### Windows
```bash
git clone https://github.com/your-username/Synopsis_Checker.git
cd Synopsis_Checker
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux
```bash
git clone https://github.com/your-username/Synopsis_Checker.git
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
### 💻 In the UI: 

1. Select LLM mode: **Local** or **ChatGPT**
2. Upload an article (`.txt` or `.pdf`) and its synopsis (`.txt`)
3. Click **Analyze** to generate scores and feedback. 

![Demo in Action](assets/demo.gif)
