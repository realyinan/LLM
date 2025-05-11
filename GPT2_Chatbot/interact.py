import os
from datetime import datetime

import torch
from transformers import GPT2LMHeadModel
from transformers import BertTokenizerFast
import torch.nn.functional as F
from parameter_config import *


PAD = '[PAD]'
pad_id = 0

def top_k_top_p_filtering(logits, top_k=0, filter_value=float("-inf")):
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))  # Safety check：确保top_k不超过logits的最后一个维度大小
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, k=top_k)[0][..., -1, None]  # 最后一维的最后一个元素，并在该位置增加一个新维度。
        logits[indices_to_remove] = filter_value
    return logits


def main():
    pconf = ParameterConfig()
    tokenizer = BertTokenizerFast(vocab_file=pconf.vocab_path)
    model = GPT2LMHeadModel.from_pretrained("./save_model/epoch97")
    model = model.to(pconf.device)
    model.eval()

    # 保存聊天记录的文件路径
    if pconf.save_samples_path:
        if not os.path.exists(pconf.save_samples_path):
            os.makedirs(pconf.save_samples_path)
        samples_file = open(pconf.save_samples_path + '/samples.txt', 'a', encoding='utf-8')
        samples_file.write("聊天记录{}:\n".format(datetime.now()))
    # 存储聊天记录，每个utterance以token的id的形式进行存储
    history = []
    print('你好，我是你的生活助手小美')

    while True:
        try:
            text = input("user: ")
            if pconf.save_samples_path:
                samples_file.write("user:{}\n".format(text))
            text_ids = tokenizer.encode(text, add_special_tokens=False)
            history.append(text_ids)
            input_ids = [tokenizer.cls_token_id]  # 每个input以[CLS]为开头

            for history_id, history_utr in enumerate(history[-pconf.max_history_len:]):
                input_ids.extend(history_utr)
                input_ids.append(tokenizer.sep_token_id)
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=pconf.device)
            input_ids = input_ids.unsqueeze(0)

            response = []  # 根据context，生成的response
            for _ in range(pconf.max_len):
                outputs = model(input_ids)
                logits = outputs.logits
                next_token_logits = logits[0, -1, :]
                for idx in set(response):
                    next_token_logits[idx] /= pconf.repetition_penalty
                next_token_logits[tokenizer.convert_tokens_to_ids("[UNK]")] = -float("inf")  # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=pconf.topk)
                # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

                if next_token.item() == tokenizer.sep_token_id:
                    break
                response.append(next_token.item())
                input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
            history.append(response)
            text = tokenizer.convert_ids_to_tokens(response)
            print("chatbot: " + "".join(text))

        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    main()