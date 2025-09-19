from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_unstructured.document_loaders import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os 
from fastapi import FastAPI, UploadFile, File, Form, Request
from tempfile import NamedTemporaryFile
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI 
app = FastAPI()
static_path = os.path.join(os.path.dirname(__file__), "static")
print("Serving static files from:", static_path)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
CHROMA_PATH = "chroma"
TEMPLATE = """
Ответь на вопрос, опираясь только на следующий контекст 
{context}
---
Ответь на вопрос, используя исключительно приведённый выше контекст:
{question}
"""
def get_file_type(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    return ext
def load_document(filepath: str):
    ext = get_file_type(filepath)
    #было бы лучше использовать unstructured
    if ext == ".pdf":
        return PyPDFLoader(filepath).load()
    elif ext == ".txt":
        return TextLoader(filepath).load()
    elif ext in [".doc", ".docx"]:
        return UnstructuredWordDocumentLoader(filepath).load()
    else:
        raise ValueError(f"Документ должен быть в формате pdf/txt/doc/docx: {ext}")
def split_docs(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    return chunks
@app.get("/")
async def root():
    return FileResponse("static/index.html")
@app.get("/chat")
async def get_chat():
    return FileResponse("static/chat.html")
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    print("one two")
    _, ext = os.path.splitext(file.filename)
    with NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

       
    docs = load_document(tmp_path)
    chunks = split_docs(docs)
 
    db = Chroma.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-small"), persist_directory=CHROMA_PATH)
    print ("Saved all chunks")
    return {"message": f"File '{file.filename}' uploaded successfully!"}

@app.post("/ask")
async def answer(request: Request):
    data = await request.json()
    question_text = data.get("question")
    #база данных
    vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))
    #поиск 
    fournn = vectordb.similarity_search_with_relevance_scores(question_text, k=4)
    context_text = "\n\n --- \n\n".join([doc.page_content for doc, _score in fournn])
    prompt_template = ChatPromptTemplate.from_template(TEMPLATE)
    prompt = prompt_template.format(context=context_text, question = question_text)
    model = ChatOpenAI(model="gpt-5-nano")
    response_text = model.predict(prompt)
    top_doc, _score = fournn[0]
    source_text = top_doc.page_content
    return {
    "answer": response_text,
    "sources": source_text
}
    

