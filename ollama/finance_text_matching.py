from rich import print
import ollama


# 提供相似，不相似的语义匹配例子
examples = {
    '是': [
        ('公司ABC发布了季度财报，显示盈利增长。', '财报披露，公司ABC利润上升。'),
    ],
    '不是': [
        ('黄金价格下跌，投资者抛售。', '外汇市场交易额创下新高。'),
        ('央行降息，刺激经济增长。', '新能源技术的创新。')
    ]
}

def init_prompts():
    pre_history = [{"role": "system", "content": "现在你需要帮助我完成文本匹配任务，当我给你两个句子时，你需要回答我这两句话语义是否相似。只需要回答是否相似，不要做多余的回答。"}]
    for key, sentence_pairs in examples.items():
        for sentence_pair in sentence_pairs:
            sentence1, sentence2 = sentence_pair
            pre_history.append({"role": "user", "content": f"句子一: {sentence1}\n句子二: {sentence2}\n上面两句话是相似的语义吗?"})
            pre_history.append({"role": "assistant", "content": key})
    return {"pre_history": pre_history}


def inference(sentence_pairs: list, custom_settings: dict):
    for sentence_pair in sentence_pairs:
        sentence1, sentence2 = sentence_pair
        sentence_with_prompt = f"句子一: {sentence1}\n句子二: {sentence2}\n句子二\n上面两句话是相似的语义吗?"
        response = ollama.chat(
            model="qwen2.5:7b",
            messages=[*custom_settings["pre_history"], {"role": "user", "content": sentence_with_prompt}]
        )
        response = response["message"]["content"]
        print("sentence_pair: ", sentence_pair)
        print("answer: ", response)

if __name__ == '__main__':
    sentence_pairs = [
        ('股票市场今日大涨，投资者乐观。', '持续上涨的市场让投资者感到满意。'),
        ('油价大幅下跌，能源公司面临挑战。', '未来智能城市的建设趋势愈发明显。'),
        ('利率上升，影响房地产市场。', '高利率对房地产有一定冲击。'),
    ]

    custom_settings = init_prompts()
    inference(sentence_pairs, custom_settings)
