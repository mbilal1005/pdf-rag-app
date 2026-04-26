# PDF RAG App

Ask questions about any PDF document using LangChain and OpenAI.

## How it works

1. Loads a PDF and splits it into small chunks
2. Converts chunks into embeddings (vectors) using OpenAI
3. Stores them in a local FAISS vector database
4. When you ask a question, finds the most relevant chunks and sends them to GPT

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
```

## Usage

```bash
python main.py path/to/your.pdf
```

Then type your questions in the terminal.

## Example

```
Your question: What is the main topic of this document?
Answer: The document is about ...
Sources used: 4 chunk(s)
  [1] Page 2: ...
```
