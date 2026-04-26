"""
Simple RAG (Retrieval-Augmented Generation) Application
Ask questions about any PDF document using LangChain + OpenAI
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
import sys


def load_and_index_pdf(pdf_path: str):
    """Load a PDF and split it into chunks, then store in a vector DB."""
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split into smaller chunks so the LLM can process them
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # Create embeddings and store in FAISS vector store
    print("Creating vector store...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def build_qa_chain(vectorstore):
    """Build a retrieval QA chain from the vector store."""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )
    return chain


def main():
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: Set your OPENAI_API_KEY environment variable first.")
        print("  export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    # Get PDF path from argument or use default
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "document.pdf"

    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' not found.")
        print("Usage: python main.py path/to/your.pdf")
        sys.exit(1)

    # Load, index and build chain
    vectorstore = load_and_index_pdf(pdf_path)
    chain = build_qa_chain(vectorstore)

    print("\nRAG app ready! Ask questions about your document.")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        result = chain.invoke({"query": question})
        print(f"\nAnswer: {result['result']}")
        print(f"\nSources used: {len(result['source_documents'])} chunk(s)")
        for i, doc in enumerate(result["source_documents"], 1):
            page = doc.metadata.get("page", "?")
            print(f"  [{i}] Page {page + 1}: {doc.page_content[:100]}...")
        print()


if __name__ == "__main__":
    main()
