# pdf-rag-app

Ask questions about any PDF document using natural language — powered by a RAG (Retrieval-Augmented Generation) pipeline.

## What it does

Most documents are too long to paste into a language model. This app solves that: it splits a PDF into chunks, embeds them into a vector store, and at query time retrieves only the most relevant sections before asking the LLM. The result is accurate, grounded answers with source page numbers.

## How it works

```
PDF file
   ↓
Split into 500-token chunks (50-token overlap)
   ↓
Embed with OpenAI → store in FAISS vector index
   ↓
Your question → retrieve top-4 matching chunks
   ↓
GPT-3.5-turbo generates a grounded answer
```

## Stack

- **LangChain** — RAG pipeline and document loading
- **OpenAI API** — embeddings + GPT-3.5-turbo for answers
- **FAISS** — local vector store for semantic search
- **PyPDF** — PDF text extraction

## Setup

```bash
git clone https://github.com/mbilal1005/pdf-rag-app
cd pdf-rag-app

pip install -r requirements.txt

export OPENAI_API_KEY="sk-..."
```

## Usage

```bash
python main.py path/to/your.pdf
```

Then ask questions in the terminal:

```
Your question: What is the main topic of this document?
Answer: The document is about ...

Sources used: 4 chunk(s)
  [1] Page 2: ...
  [2] Page 5: ...
```

Type `quit` to exit.
