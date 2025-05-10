from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_community.agent_toolkits.load_tools import load_tools


# 实例化大模型
llm = OllamaLLM(model="qwen2.5:7b")

# 设置工具
tools = load_tools(tool_names=["llm-math"], llm=llm)

# 实例化代理Agent, 返回AgentExecutor
agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# 准备提示提
prompt_template = """解以下方程：3x + 4(x + 2) - 84 = y; 其中x为3，请问y是多少？"""
prompt = PromptTemplate.format_prompt(prompt_template)

# 代理Agent工作
result = agent.invoke(prompt)
print(result)
