from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM


template = "我的邻居姓{lastname}, 请帮他孩子起一个名字"

prompt = PromptTemplate(
    input_variables=["lastname"],
    template=template
)

llm = OllamaLLM(model="qwen2.5:7b")

chain = prompt | llm

print(chain.invoke({"lastname": "刘"}))