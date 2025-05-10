from langchain.chains import ConversationChain
from langchain_ollama.llms import OllamaLLM


llm = OllamaLLM(model="qwen2.5:7b")

conversation = ConversationChain(llm=llm)
result1 = conversation.predict(input="小明有一只猫")
print(result1)
print('*'*80)
resutl2 = conversation.predict(input="小刚有2只狗")
print(resutl2)
print('*'*80)
resutl3 = conversation.predict(input="小明和小刚一共有几只宠物?")
print(resutl3)
print('*'*80)