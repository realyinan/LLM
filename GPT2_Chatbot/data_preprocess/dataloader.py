from torch.utils.data import DataLoader
from dataset import *
import pickle
from parameter_config import *
import torch.nn.utils.rnn as rnn_utils  # 导入rnn_utils模块，用于处理可变长度序列的填充和排序


params = ParameterConfig()

def load_dataset(train_path, valid_path):
    with open(train_path, "rb") as f:
        train_input_list = pickle.load(f)
    with open(valid_path, "rb") as f:
        valid_input_list = pickle.load(f)

    train_dataset = MyDataset(train_input_list, params.max_len)  # 创建训练数据集对象
    valid_dataset = MyDataset(valid_input_list, params.max_len)  # 创建验证数据集对象
    return train_dataset, valid_dataset

def collate_fn(batch):
    input_ids = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)  #  rnn_utils.pad_sequence：将根据一个batch中，最大句子长度，进行补齐
    labels = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=-100)  # 对标签序列进行填充，使其长度一致
    return input_ids, labels



def get_dataloader(train_path, valid_path):
    train_dataset, valid_dataset = load_dataset(train_path, valid_path)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True)
    return train_dataloader, valid_dataloader


if __name__ == '__main__':
    train_dataloader, valid_dataloader = get_dataloader(params.train_path, params.valid_path)
    for input_ids, labels in valid_dataloader:
        print(f'input_ids--》{input_ids.shape}')
        print(f'labels--》{labels.shape}')
        break
