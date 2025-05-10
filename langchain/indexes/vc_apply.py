from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma


with open("./pku.txt", "r", encoding="utf-8") as f:
    state_of_the_union = f.read()
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_text(state_of_the_union)
print(texts)
print(len(texts))

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
docsearch = Chroma.from_texts(texts, embeddings)

query = "1937年北京发生了什么?"
docs = docsearch.similarity_search(query)
print(docs)
print(len(docs))