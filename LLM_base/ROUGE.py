from rouge import Rouge


generated_text = "This is some generated text."
reference_texts = "This is another generated reference text."

#计算ROUGE指标
rouge = Rouge()
scores = rouge.get_scores(generated_text, reference_texts)

print(scores)
