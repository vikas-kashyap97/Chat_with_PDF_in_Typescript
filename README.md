# Chat with PDFs using LangChain + OpenAI


This project lets you **upload a PDF, chunk its contents, embed them, and query it** using OpenAI’s `gpt-3.5-turbo` model with retrieval-augmented generation (RAG).  
It’s built with **TypeScript, LangChain, and ts-node**.


---

## 🚀 Features
- Load and parse PDFs with `@langchain/community`’s `PDFLoader`
- Split documents into manageable chunks with `RecursiveCharacterTextSplitter`
- Create embeddings using OpenAI’s `text-embedding-3-large`
- Store embeddings in an **in-memory vector store** (`MemoryVectorStore`)
- Query the PDF using OpenAI’s `gpt-3.5-turbo` with `RetrievalQAChain`

---

