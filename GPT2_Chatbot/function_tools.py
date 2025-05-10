import torch
import torch.nn.functional as F

def calculate_acc(logits, labels, ignore_index=-100):
    # print(f'logits--->原始值的形状{logits.shape}')
    # print(f'labels--->原始值的形状{labels.shape}')
    # print(f'logits.size-->{logits.size(-1)}')
    # print(f'logit[:, :-1, :]-->{logits[:, :-1, :].shape}')
    logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
    # print(f'logits改变完形状的--->{logits.shape}')
    # print(f'labels[:, 1:]--->{labels[:, 1:].shape}')
    labels = labels[:, 1:].contiguous().view(-1)
    # print(f'labels改变完之后的形状--->{labels.shape}')
    _, predicts = logits.max(dim=-1)
    # print(f'predicts预测结果---》{predicts.shape}')
    '''
    在 PyTorch 中，labels.ne(ignore_index) 表示将标签张量 labels 中的值不等于 ignore_index 的位置标记为 True，等于 ignore_index 的位置标记为 False。
    这个操作，以过滤掉 ignore_index 对损失的贡献
    '''
    non_pad_mask = labels.ne(ignore_index)
    # print(f'non_pad_mask-->{non_pad_mask}')
    '''
    在 PyTorch 中，logit.eq(labels) 表示将模型的预测输出值 logit 中等于标签张量 labels 的位置标记为 True，
    不等于标签张量 labels 的位置标记为 False。以标记出预测输出值和标签值相等的位置。
    masked_select(non_pad_mask) 表示将张量中非填充标记的位置选出来。
    '''
    n_correct = predicts.eq(labels).masked_select(non_pad_mask).sum().item()
    # print(f'n_correct-->{n_correct}')
    n_word = non_pad_mask.sum().item()
    return n_correct, n_word



