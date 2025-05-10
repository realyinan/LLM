from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM

model = OllamaLLM(model="qwen2.5:7b")

#定义模板
template = "我的邻居姓{lastname}, 他妻子姓{lastname2}, 请帮他孩子起一个名字"

prompt = PromptTemplate(
    input_variables=["lastname", "lastname2"],
    template=template
)

prompt_text = prompt.format(lastname="李", lastname2="刘")
print(prompt_text)

result = model.invoke(prompt_text)
print(result)