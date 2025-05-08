from langchain_ollama.embeddings import OllamaEmbeddings

model = OllamaEmbeddings(model="mxbai-embed-large")
res1 = model.embed_query("这是一个测试文档")
print(res1)
print(len(res1))  # 1024


res2 = model.embed_documents([
    "床前明月光",
    "疑是地上霜"
])
print(res2) # [1024, 1024]
