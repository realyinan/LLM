from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_ollama import ChatOllama

model = ChatOllama(model="qwen2.5:7b")
messages = [
    SystemMessage(content="你现在是一本唐诗三百首"),
    HumanMessage(content="生成将进酒")
]

res = model.invoke(messages)
print(res.content)