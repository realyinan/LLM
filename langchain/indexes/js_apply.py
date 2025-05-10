from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

loader = TextLoader("./pku.txt", encoding="utf-8")
documents = loader.load()

text_spliter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_spliter.split_documents(documents)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db = FAISS.from_documents(texts, embeddings)
retriver = db.as_retriever(search_kwargs={"k": 1})
docs = retriver.get_relevant_documents("北京大学什么适合成立")
print(docs)