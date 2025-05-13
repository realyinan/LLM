import torch
import numpy as np
from pprint import pprint
from datasets import load_dataset
from transformers import AutoTokenizer
from ptune_config import *
from functools import partial


def convert_example(
        examples: dict,
        tokenizer,
        max_seq_len: int,
        max_label_len: int,
        p_embedding_num=6,
        train_mode=True,
        return_tensor=False
) -> dict:
    """
    将样本数据转换为模型接收的输入数据。

    Args:
        examples (dict): 训练数据样本, e.g. -> {
                                                "text": [
                                                            '娱乐	嗨放派怎么停播了',
                                                            '体育	世界杯为何迟迟不见宣传',
                                                            ...
                                                ]
                                            }
        tokenizer:
        max_seq_len:
        max_label_len (int): 最大label长度，若没有达到最大长度，则padding为最大长度
        p_embedding_num (int): p-tuning token 的个数
        train_mode (bool): 训练阶段 or 推理阶段。
        return_tensor (bool): 是否返回tensor类型，如不是，则返回numpy类型。

    Returns:
        dict (str: np.array) -> tokenized_output = {
                            'input_ids': [[101, 3928, ...], [101, 4395, ...]],
                            'token_type_ids': [[0, 0, ...], [0, 0, ...]],
                            'mask_positions': [[5, 6, ...], [3, 4, ...]],
                            'mask_labels': [[183, 234], [298, 322], ...]
                        }
    """
    tokenized_output = {
        'input_ids': [],
        'attention_mask': [],
        'mask_positions': [],  # 记录label的位置（即MASK Token的位置）
        'mask_labels': []  # 记录MASK Token的原始值（即Label值）
    }

    for i, example in enumerate(examples["text"]):
        try:
            start_mask_position = 1

            if train_mode:
                label, content = example.strip().split("\t", 1)
            else:
                content = example.strip()

            encoded_inputs = tokenizer(
                text=content,
                truncation=True,
                max_length=max_seq_len,
                padding="max_length"
            )
        except Exception as e:
            continue

        input_ids = encoded_inputs["input_ids"]

        mask_tokens = ["[MASK]"] * max_label_len
        mask_ids = tokenizer.convert_tokens_to_ids(mask_tokens)

        p_tokens = ["[unused{}]".format(i+1) for i in range(p_embedding_num)]
        p_tokens_ids = tokenizer.convert_tokens_to_ids(p_tokens)

        tmp_input_ids = input_ids[:-1]  # 59
        tmp_input_ids = tmp_input_ids[:max_seq_len - len(mask_ids) - len(p_tokens) - 1]  # 最大长度-p_token长度-label长度，51
        tmp_input_ids = tmp_input_ids[:start_mask_position] + mask_ids + tmp_input_ids[start_mask_position:]  # 53
        input_ids = tmp_input_ids + [input_ids[-1]]  # 54
        input_ids = p_tokens_ids + input_ids  # 60
        tokenized_output["input_ids"].append(input_ids)
        # print("input_ids-->", input_ids)

        mask_positions = [len(p_tokens_ids) + start_mask_position + i for i in range(max_label_len)]
        tokenized_output["mask_positions"].append(mask_positions)
        # print("mask_positions-->", mask_positions)

        if 'token_type_ids' in encoded_inputs:  # 兼容不需要 token_type_id 的模型, e.g. Roberta-Base
            tmp = encoded_inputs['token_type_ids']
            if 'token_type_ids' not in tokenized_output:
                tokenized_output['token_type_ids'] = [tmp]
            else:
                tokenized_output['token_type_ids'].append(tmp)

        attention_mask = get_attention_mask(input_ids)
        tokenized_output["attention_mask"].append(attention_mask)

        if train_mode:
            mask_labels = tokenizer(text=label)
            mask_labels = mask_labels['input_ids'][1:-1]
            mask_labels = mask_labels[:max_label_len]
            mask_labels += [tokenizer.pad_token_id] * (max_label_len - len(mask_labels))  # 将 label 补到最长
            tokenized_output['mask_labels'].append(mask_labels)

    for k, v in tokenized_output.items():
        if return_tensor:
            tokenized_output[k] = torch.LongTensor(v)
        else:
            tokenized_output[k] = np.array(v)

    return tokenized_output


def get_attention_mask(alist):
    new_a = np.where(np.array(alist)>0, 1, 0)
    return new_a.tolist()



if __name__ == '__main__':
    pc = ProjectConfig()
    train_dataset = load_dataset("text", data_files={"train": pc.train_path})
    print("train_dataset-->", train_dataset)
    tokenizer = AutoTokenizer.from_pretrained(pc.pre_model)
    # res = convert_example(
    #     train_dataset["train"],
    #     tokenizer=tokenizer,
    #     max_seq_len=60,
    #     max_label_len=2,
    #     p_embedding_num=6
    # )

    new_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_len=pc.max_seq_len,
        max_label_len=pc.max_label_len,
        p_embedding_num=pc.p_embedding_num
    )

    dataset = train_dataset.map(new_func, batched=True)
    print("dataset-->", dataset)