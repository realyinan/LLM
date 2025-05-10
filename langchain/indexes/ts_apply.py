from langchain.text_splitter import CharacterTextSplitter


text_splitter = CharacterTextSplitter(
    separator=" ", # 空格分割
    chunk_size=5,
    chunk_overlap=1
)

# 一句话分割
a = text_splitter.split_text("a b c d e f")
print(a)

# 多句话分割
texts = text_splitter.create_documents(["a b c d e f", "e f g h"],)
print(texts)