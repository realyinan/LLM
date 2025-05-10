from langchain_community.document_loaders import TextLoader
loader = TextLoader(file_path="./衣服属性.txt", encoding="utf-8")
docs = loader.load()
print(docs)
print(len(docs))
first_01 = docs[0].page_content[:10]
print(first_01)