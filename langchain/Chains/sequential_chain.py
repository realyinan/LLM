from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

llm = OllamaLLM(model="qwen2.5:7b")
parser = StrOutputParser()


prompt = PromptTemplate(
    input_variables=["lastname"],
    template="我的邻居姓{lastname}, 请帮他孩子起一个名字 (别解释, 直接生成一个名字就行了)"
)

chain1 = prompt | llm | parser

prompt1 = PromptTemplate(
    input_variables=["child_name"],
    template="我的邻居的小孩叫{child_name}, 请帮他孩子起一个小名"
)

chain2 = prompt1 | llm | parser

child_name = chain1.invoke({"lastname": "高"}).strip()

nick_name = chain2.invoke({"child_name": child_name}).strip()

print("child_name: ", child_name)
print("nick_name: ", nick_name)


