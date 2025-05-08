from langchain_ollama import OllamaLLM


model = OllamaLLM(model="qwen2.5:7b")
result = model.invoke("介绍一下费德勒")
print(result)