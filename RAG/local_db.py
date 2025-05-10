from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from rich import print


def get_vector():
    # 第一步加载文档
    loader = PyMuPDFLoader("./物流信息.pdf")

    # 将文本转成 Document 对象
    data = loader.load()
    print(data)

    # 切分文本
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20)

    # 切割加载的document
    split_docs = text_spliter.split_documents(data)
    # print(len(split_docs))
    print(split_docs)

    # 初始化embedding
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    # 将document通过embeddings对象计算得到向量信息并永久存入FAISS向量数据库, 用于后续匹配查询
    db = FAISS.from_documents(split_docs, embeddings)
    db.save_local("./faiss/wuliu")


if __name__ == '__main__':
    get_vector()