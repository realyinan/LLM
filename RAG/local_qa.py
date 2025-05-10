import time
from local_db import *
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM


embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db = FAISS.load_local("./faiss/wuliu", embeddings, allow_dangerous_deserialization=True)

start_time = time.time()


def get_related_content(related_docs):
    related_content = []
    for doc in related_docs:
        related_content.append(doc.page_content)

    return "\n".join(related_content)


def define_prompt():
    question = "我的快递出发地是哪里? 预计几天的时间到达?"
    docs = db.similarity_search(question, k=2)
    related_content = get_related_content(docs)

    PROMPT_TEMPLATE = """
    基于以下已知信息, 简洁专业的来回答用户问题, 不允许有添加和编造的成分
    已知内容:
    {context}
    问题:
    {question}"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE
    )

    my_pmt = prompt.format(context=related_content, question=question)

    return my_pmt

def qa():
    model = OllamaLLM(model="qwen2.5:7b")
    my_pmt = define_prompt()
    result = model.invoke(my_pmt)
    return result



if __name__ == '__main__':
    result = qa()
    print(result)
    end_time = time.time()
    print(end_time-start_time)
