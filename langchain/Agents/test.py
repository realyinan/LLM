from langchain_community.agent_toolkits.load_tools import get_all_tool_names

result = get_all_tool_names()
print(result)
print(len(result))