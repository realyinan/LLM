from torch.utils.data import Dataset
import torch
import pickle


class MyDataset(Dataset):
    def __init__(self, input_list, max_len):
        super().__init__()
        self.input_list = input_list   # 将输入列表赋值给数据集的input_list属性
        self.max_len = max_len  # 将最大序列长度赋值给数据集的max_len属性

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, item):
        input_ids = self.input_list[item]
        input_ids = input_ids[:self.max_len]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids


if __name__ == '__main__':
    with open("../data/medical_train.pkl", "rb") as f:
        train_input_list = pickle.load(f)
    mydatase = MyDataset(input_list=train_input_list, max_len=300)
    print(len(mydatase))
    print(mydatase[0])