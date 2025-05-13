from torch.utils.data import DataLoader
from transformers import default_data_collator
from data_preprocess import *
from pet_config import *


pc = ProjectConfig()
tokenizer = AutoTokenizer.from_pretrained(pc.pre_model)

def get_data():
    prompt = open(pc.prompt_file, "r", encoding="utf-8").read().strip()
    hard_template = HardTemplate(prompt=prompt)
    dataset = load_dataset("text", data_files={
        "train": pc.train_path,
        "dev": pc.dev_path
    })

    new_func = partial(convert_example,
                           tokenizer=tokenizer,
                           hard_template=hard_template,
                           max_seq_len=pc.max_seq_len,
                           max_label_len=pc.max_label_len)
    dataset = dataset.map(new_func, batched=True)

    train_dataset = dataset["train"]
    dev_dataset = dataset["dev"]

    train_dataloader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=pc.batch_size,
        collate_fn=default_data_collator
    )
    dev_dataloader = DataLoader(
        dataset=dev_dataset,
        shuffle=True,
        batch_size=pc.batch_size,
        collate_fn=default_data_collator
    )

    return train_dataloader, dev_dataloader

if __name__ == '__main__':
    train_dataloader, dev_dataloader = get_data()
    for i, value in enumerate(train_dataloader):
        print(value)
        break