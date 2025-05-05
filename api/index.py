from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import WebBaseLoader
import os

app = FastAPI()

# Load once on startup
@app.on_event("startup")
async def load_index():
    global qa_chain
    loader = WebBaseLoader("https://skytechsolutions.us")  # Replace with your real site
    docs = loader.load()
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question")
    if not question:
        return JSONResponse({"error": "No question provided"}, status_code=400)
    
    answer = qa_chain.run(question)
    return {"answer": answer}
