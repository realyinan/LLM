from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_ollama.llms import OllamaLLM

model = OllamaLLM(model="qwen2.5:7b")

examples = [
    {"word": "开心", "antonym": "难过"},
    {"word": "高", "antonym": "矮"}
]

example_template = """
单词: {word}
反义词: {antonym}\n"""

example_prompt = PromptTemplate(
    input_variables=["words", "antonym"],
    template=example_template
)

few_shot_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="给出每个单词的反义词",
    suffix="单词: {input}\n反义词:",
    input_variables=["input"],
    example_separator="\n"
)

prompt_text = few_shot_template.format(input="大")

# print(prompt_text)

print(model.invoke(prompt_text))