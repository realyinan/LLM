from torch.utils.data import DataLoader
from transformers import default_data_collator, AutoTokenizer
from ptune_config import *
from data_preprocess import *


pc = ProjectConfig()
tokenizer = AutoTokenizer.from_pretrained(pc.pre_model)

def get_data():
    dataset = load_dataset("text", data_files={"train": pc.train_path, "dev": pc.dev_path})

    new_func = partial(convert_example,
                       tokenizer=tokenizer,
                       max_seq_len=pc.max_seq_len,
                       max_label_len=pc.max_label_len,
                       p_embedding_num=pc.p_embedding_num)

    dataset = dataset.map(new_func, batched=True)
    train_dataset = dataset["train"]
    dev_dataset = dataset["dev"]

    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  collate_fn=default_data_collator,
                                  batch_size=pc.batch_size)
    dev_dataloader = DataLoader(dev_dataset,
                                collate_fn=default_data_collator,
                                batch_size=pc.batch_size)
    return train_dataloader, dev_dataloader

if __name__ == '__main__':
    train_dataloader, dev_dataloader = get_data()
    for batch in train_dataloader:
        print(batch)
        break