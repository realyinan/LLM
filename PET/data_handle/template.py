from rich import print
from transformers import AutoTokenizer
import numpy as np
from pet_config import *


class HardTemplate(object):
    """
    硬模板，人工定义句子和[MASK]之间的位置关系。
    """
    def __init__(self, prompt: str):
        self.prompt = prompt
        self.inputs_list = []  # 根据文字prompt拆分为各part的列表
        self.custom_tokens = set(['MASK'])  # 从prompt中解析出的自定义token集合
        self.prompt_analysis()  # 解析prompt模板

    def prompt_analysis(self):
        """
        将prompt文字模板拆解为可映射的数据结构。
        Examples:
            prompt -> "这是一条{MASK}评论：{textA}。"
            inputs_list -> ['这', '是', '一', '条', 'MASK', '评', '论', '：', 'textA', '。']
            custom_tokens -> {'textA', 'MASK'}
        """
        idx = 0
        while idx < len(self.prompt):
            str_part = ""
            if self.prompt[idx] not in ["{", "}"]:
                self.inputs_list.append(self.prompt[idx])

            if self.prompt[idx] == "{":
                idx += 1
                while self.prompt[idx] != "}":
                    str_part += self.prompt[idx]
                    idx += 1
            elif self.prompt[idx] == "}":
                raise ValueError("Unmatched bracket '}', check your prompt.")

            if str_part:
                self.inputs_list.append(str_part)
                self.custom_tokens.add(str_part)
            idx += 1

    def __call__(self, inputs_dict: dict, tokenizer, mask_length, max_seq_length=512):
        """
        输入一个样本，转换为符合模板的格式。
        Args:
            inputs_dict (dict): prompt中的参数字典, e.g. -> {
                                                            "textA": "这个手机也太卡了",
                                                            "MASK": "[MASK]"
                                                        }
            tokenizer: 用于encoding文本
            mask_length (int): MASK token 的长度

        Returns:
            dict -> {
                'text': '[CLS]这是一条[MASK]评论：这个手机也太卡了。[SEP]',
                'input_ids': [1, 47, 10, 7, 304, 3, 480, 279, 74, 47, 27, 247, 98, 105, 512, 777, 15, 12043, 2],
                'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                'mask_position': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        """





if __name__ == "__main__":
    pc = ProjectConfig()
    tokenizer = AutoTokenizer.from_pretrained(pc.pre_model)
    hard_template = HardTemplate(prompt="这是一条{MASK}评论：{textA}。")
    print(hard_template.inputs_list)
    print(hard_template.custom_tokens)


