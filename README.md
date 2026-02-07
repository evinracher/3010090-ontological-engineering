# README

## Course Information
- **Program:** Specialization in Artificial Intelligence
- **SNIES Code:** 108149
- **University:** Universidad Nacional de Colombia
- **Faculty:** Facultad de Minas
- **Course Name:** Ingeniería Ontológica
- **Course Code (SIA):** 3010090

## Repository Purpose
This repository contains academic exercises and workshops from the Artificial Intelligence Specialization. The work focuses on practical experimentation with NLP and multimodal AI concepts, combining theoretical foundations with hands-on implementations in notebooks.

## Content Overview
- Main topics: tokenization, embeddings, transformer models, LLM interaction patterns, prompt engineering, and multimodal models.
- Artifacts included: Jupyter notebooks (`Taller_1.ipynb`, `Taller_2.ipynb`) and code cells with experiments and demonstrations.
- High-level structure: `Taller_1.ipynb` covers tokenization, embeddings, transformers, and BERT-based tasks; `Taller_2.ipynb` covers LLM API connections, prompt engineering, and multimodal pipelines (text, image, audio).

## Key Concepts Implemented
- Tokenization strategies for transformer models
- Word embeddings and semantic similarity
- Transformer architectures and pretrained models
- Sentiment analysis and named entity recognition
- LLM interaction patterns (completion, chat, multimodal)
- Prompt engineering patterns and evaluation
- Multimodal similarity (text–image) and speech-to-text
- Diffusion-based image generation

## Repository Analysis
### Packages and Libraries
- `transformers`
- `gensim`
- `torch`
- `diffusers`
- `safetensors`
- `Pillow` (PIL)
- `requests`
- `numpy`
- `openai`
- `google-genai`
- `IPython`

### Techniques and Approaches
- Text tokenization with pretrained tokenizers
- Word embedding similarity queries (analogy and nearest neighbors)
- Transformer-based text classification and NER pipelines
- LLM text generation using GPT-2
- Prompt engineering patterns for instruction design
- CLIP-based text–image similarity scoring
- Automatic speech recognition with Whisper
- Diffusion model image generation with DDPM

### Methodologies
- Interactive experimentation in notebooks
- Use of pretrained models for rapid prototyping
- Comparative exploration of multiple model families and interaction modes

## Technologies and Tools
- **Languages:** Python
- **Frameworks/Libraries:** Hugging Face Transformers, Gensim, PyTorch, Diffusers
- **Platforms/Tools:** Jupyter/Colab-style notebooks, external LLM APIs

## How to Run / Reproduce
1. Create and activate a Python environment (3.9+ recommended).
2. Install dependencies:
   ```bash
   pip install jupyter transformers gensim torch diffusers safetensors pillow requests numpy openai google-genai
   ```
3. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
4. Open `Taller_1.ipynb` or `Taller_2.ipynb` and run cells in order.
5. For API-based sections in `Taller_2.ipynb`, set required keys in your environment (e.g., `GOOGLE_API_KEY` and any provider-specific keys referenced in the notebook).

## Skills Demonstrated
- NLP preprocessing and embedding-based reasoning
- Transformer model usage for classification and NER
- Prompt engineering and LLM interaction design
- Multimodal AI experimentation (text, image, audio)
- Reproducible experimentation in notebooks

## Academic Disclaimer
> **Disclaimer:**  
> Some code comments and variable names may appear in Spanish, as the course was taught in Spanish. The README and main documentation are provided in English for broader accessibility.

## Academic Context
This repository is part of a formal academic specialization program. The code prioritizes clarity, learning, and experimentation over production-level optimization.
