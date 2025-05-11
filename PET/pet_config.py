import torch


class ProjectConfig(object):
    def __init__(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # windows电脑/linux服务器
        self.pre_model = r'C:\Users\19981\Documents\GitHub\LLM\bert-base-chinese'
        self.train_path = r'C:\Users\19981\Documents\GitHub\LLM\PET\data\train.txt'
        self.dev_path = r'C:\Users\19981\Documents\GitHub\LLM\PET\data\dev.txt'
        self.prompt_file = r'C:\Users\19981\Documents\GitHub\LLM\PET\data\prompt.txt'
        self.verbalizer = r'C:\Users\19981\Documents\GitHub\LLM\PET\data\verbalizer.txt'
        self.max_seq_len = 256
        self.batch_size = 8
        self.learning_rate = 5e-5
        self.weight_decay = 0
        self.warmup_ratio = 0.06
        self.max_label_len = 2
        self.epochs = 20
        self.logging_steps = 2
        self.valid_steps = 20
        self.save_dir = r'C:\Users\19981\Documents\GitHub\LLM\PET\checkpoints'


if __name__ == '__main__':
    pc = ProjectConfig()
    print(pc.prompt_file)
    print(pc.pre_model)
