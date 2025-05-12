import torch
import numpy as np
from template import *
from datasets import load_dataset
from functools import partial # partial是对函数进行再次封装，便于使用
from pet_config import *


def convert_example(
        examples: dict,
        tokenizer: AutoTokenizer,
        max_seq_len: int,
        max_label_len: int,
        hard_template: HardTemplate,
        train_mode=True,
        return_tensor=False) -> dict:
    """
       将样本数据转换为模型接收的输入数据。

       Args:
           examples (dict): 训练数据样本, e.g. -> {
                                                   "text": [
                                                               '手机	这个手机也太卡了。',
                                                               '体育	世界杯为何迟迟不见宣传',
                                                               ...
                                                   ]
                                               }
           tokenizer:
           max_seq_len (int): 句子的最大长度，若没有达到最大长度，则padding为最大长度
           max_label_len (int): 最大label长度，若没有达到最大长度，则padding为最大长度
           hard_template (HardTemplate): 模板类。
           train_mode (bool): 训练阶段 or 推理阶段。
           return_tensor (bool): 是否返回tensor类型，如不是，则返回numpy类型。

       Returns:
           dict (str: np.array) -> tokenized_output = {
                               'input_ids': [[1, 47, 10, 7, 304, 3, 3, 3, 3, 47, 27, 247, 98, 105, 512, 777, 15, 12043, 2], ...],
                               'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ...],
                               'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], ...],
                               'mask_positions': [[5, 6, 7, 8], ...],
                               'mask_labels': [[2372, 3442, 0, 0], [2643, 4434, 2334, 0], ...]
                           }
       """
    tokenized_output = {
        'input_ids': [],
        'token_type_ids': [],
        'attention_mask': [],
        'mask_positions': [],
        'mask_labels': []
    }

    for i, example in enumerate(examples["text"]):
        if train_mode:
            label, content = example.strip().split("\t")
        else:
            content = example.strip()

        inputs_dict = {'textA': content, 'MASK': '[MASK]'}

        encoded_inputs = hard_template(
            inputs_dict=inputs_dict,
            tokenizer=tokenizer,
            max_seq_length=max_seq_len,
            mask_length=max_label_len
        )
        tokenized_output['input_ids'].append(encoded_inputs["input_ids"])
        tokenized_output['token_type_ids'].append(encoded_inputs["token_type_ids"])
        tokenized_output['attention_mask'].append(encoded_inputs["attention_mask"])
        tokenized_output['mask_positions'].append(encoded_inputs["mask_position"])

        if train_mode:
            label_encoded = tokenizer(text=[label])
            label_encoded = label_encoded["input_ids"][0][1:-1]
            label_encoded = label_encoded[:max_label_len]
            label_encoded = label_encoded + [tokenizer.pad_token_id]*(max_label_len-len(label_encoded))

            tokenized_output["mask_labels"].append(label_encoded)

    for k, v in tokenized_output.items():
        if return_tensor:
            tokenized_output[k] = torch.LongTensor(v)
        else:
            tokenized_output[k] = np.array(v)

    return tokenized_output


if __name__ == '__main__':
    pc = ProjectConfig()
    train_dataset = load_dataset("text", data_files=pc.train_path)
    # print("train_dataset-->: ", train_dataset)
    # print(train_dataset["train"]["text"])
    tokenizer = AutoTokenizer.from_pretrained(pc.pre_model)
    hard_template = HardTemplate(prompt="这是一条{MASK}评论：{textA}")
    examples = {"text": ['手机	这个手机也太卡了。', '体育	世界杯为何迟迟不见宣传']}
    # tokenized_output = convert_example(examples=examples,
    #                                     tokenizer=tokenizer,
    #                                     max_seq_len=30,
    #                                     max_label_len=2,
    #                                     hard_template=hard_template,
    #                                     train_mode=True,
    #                                     return_tensor=False)
    # print(tokenized_output)
    convert_func = partial(convert_example,
                           tokenizer=tokenizer,
                           hard_template=hard_template,
                           max_seq_len=30,
                           max_label_len=2)
    # batched=True相当于将train_dataset看成一个批次的样本直接对数据进行处理，节省时间
    # train_dataset 中的所有数据通过 convert_func 转换后，得到新的 dataset。这个操作返回的仍然是一个 datasets.Dataset 对象，
    dataset = train_dataset.map(convert_func, batched=True)
    print(dataset)
    print(dataset["train"])
    print(len(dataset["train"]))
    print(dataset["train"][0])

