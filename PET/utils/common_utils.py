import torch
from rich import print

def mlm_loss(logits,
             mask_positions,
             sub_mask_labels,
             cross_entropy_criterion,
             device
             ):
    """
    计算指定位置的mask token的output与label之间的cross entropy loss。
    Args:
        logits (torch.tensor): 模型原始输出 -> (batch, seq_len, vocab_size)
        mask_positions (torch.tensor): mask token的位置  -> (batch, mask_label_num)
        sub_mask_labels (list): mask token的sub label, 由于每个label的sub_label数目不同，所以这里是个变长的list,
                                    e.g. -> [
                                        [[2398, 3352]],
                                        [[2398, 3352], [3819, 3861]]
                                    ]
        cross_entropy_criterion (CrossEntropyLoss): CE Loss计算器
        device (str): cpu还是gpu

    Returns:
        torch.tensor: CE Loss
    """
    batch_size, seq_len, vocab_size = logits.size()
    loss = None

    for single_value in zip(logits, sub_mask_labels, mask_positions):
        single_logits = single_value[0]  # [256, 21128]
        # print(f'single_logits-->{single_logits.shape}')

        single_sub_mask_labels = single_value[1]  # [[1234, 5463]. [2356, 8745]]
        # print(f'single_sub_mask_labels-->{single_sub_mask_labels}')

        single_mask_positions = single_value[2]  # [[5, 6]]
        # print(f'single_mask_positions-->{single_mask_positions}')

        single_mask_logits = single_logits[single_mask_positions]  # [2, 21128]
        # print(f'single_mask_logits--<{single_mask_logits.shape}')

        single_mask_logits = single_mask_logits.repeat(len(single_sub_mask_labels), 1, 1)  # [2, 2, 21128]
        # print(f'single_mask_logits:{single_mask_logits.shape}')
        single_mask_logits = single_mask_logits.reshape(-1, vocab_size)  # [4, 21128]

        single_sub_mask_labels = torch.LongTensor(single_sub_mask_labels).to(device)  # [2, 2]
        single_sub_mask_labels = single_sub_mask_labels.reshape(-1, 1).squeeze()  # [4,]

        cur_loss = cross_entropy_criterion(single_mask_logits, single_sub_mask_labels)
        cur_loss = cur_loss / len(single_sub_mask_labels)

        if not loss:
            loss = cur_loss
        else:
            loss += cur_loss
    loss = loss / batch_size
    return loss


def convert_logits_to_ids(logits: torch.tensor, mask_positions: torch.tensor):
    """
    输入Language Model的词表概率分布（LMModel的logits），将mask_position位置的
    token logits转换为token的id。
    Args:
        logits (torch.tensor): model output -> (batch, seq_len, vocab_size)[2, 256, 21128]
        mask_positions (torch.tensor): mask token的位置 -> (batch, mask_label_num)[2,2]

    Returns:
        torch.LongTensor: 对应mask position上最大概率的推理token -> (batch, mask_label_num)[2, 2]
    """
    label_length = mask_positions.size()[1]  # 标签长度
    batch_size, seq_len, vocab_size = logits.size()

    mask_positions_after_reshaped = []

    for batch, mask_pos in enumerate(mask_positions.numpy().tolist()):
        for pos in mask_pos:
            mask_positions_after_reshaped.append(batch * seq_len + pos)
    # print(f'mask_positions_after_reshaped-->{mask_positions_after_reshaped}')
    logits = logits.reshape(batch_size * seq_len, -1)  # [2*256, 21128]

    mask_logits = logits[mask_positions_after_reshaped]  # [4, 21128]
    # print('选择真实掩码位置预测的数据形状',mask_logits.shape)

    predict_tokens = mask_logits.argmax(dim=-1)  # [4,]
    # print('求出每个样本真实mask位置预测的tokens', predict_tokens)

    predict_tokens = predict_tokens.reshape(-1, label_length)  # [2, 2]

    return predict_tokens





