import torch
import torch.optim as optim
import os
from rich import print
from datetime import datetime
import transformers
from transformers import GPT2LMHeadModel, GPT2Config
from transformers import BertTokenizerFast
from function_tools import *
from parameter_config import *
from data_preprocess.dataloader import *
from tqdm import tqdm


def train_epoch(model, train_dataloader, optimizer, scheduler, epoch, params):
    model.train()
    ignore_index = params.ignore_index
    epoch_start_time = datetime.now()
    total_loss = 0  # 记录下整个epoch的loss总和
    epoch_correct_num, epoch_total_num = 0, 0  # 预测正确的个数, 总的个数

    for batch_index, (input_ids, labels) in enumerate(tqdm(train_dataloader), start=1):
        input_ids = input_ids.to(params.device)
        labels = labels.to(params.device)
        outputs = model(input_ids, labels=labels)
        logits = outputs.logits
        loss = outputs.loss
        # 统计该batch预测token的正确数与总数
        batch_correct_num, batch_total_num = calculate_acc(logits, labels, ignore_index=ignore_index)
        # 计算该batch的accuracy
        batch_acc = batch_correct_num / batch_total_num
        epoch_correct_num += batch_correct_num
        epoch_total_num += batch_total_num
        total_loss += loss.item()
        





def train(model, train_dataloader, valid_dataloader, params):
    # t_total模型训练完毕，一共要迭代多少步
    t_total = len(train_dataloader) // params.gradient_accumulation_steps * params.epochs
    # eps，为了增加数值计算的稳定性而加到分母里的项，其为了防止在实现中除以零
    optimizer = optim.AdamW(model.parameters(), lr=params.lr, eps=params.eps)
    """
    学习率预热的目的是让模型在初始阶段更快地适应数据，避免训练过程中因为学习率过大导致的梯度爆炸等问题，从而提高模型的训练效果和泛化性能。
    get_linear_schedule_with_warmup：学习率从0线性（也可非线性）增加到优化器中的初始预设lr，之后使其学习率从优化器中的初始lr线性降低到0
    optimizer：这个参数需要传入一个优化器对象（optimizer object）。它代表在训练过程中用于更新模型参数的优化器，比如Adam或SGD等。
    num_warmup_steps：这个参数确定学习率在开始阶段从0线性增加到初始值的步数。在Transformer模型中，通过逐渐增加学习率来稳定和加速训练过程是常见的做法。通常，这个值是总训练步数的一小部分。
    num_training_steps：这个参数指定了总的训练步数或迭代次数。它表示优化器将在给定数据集上进行多少次参数更新。
    """
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=params.warmup_steps, num_training_steps=t_total
    )

    print('starting training')

    # 用于记录每个epoch训练和验证的loss
    train_losses, validate_losses = [], []
    # 记录验证集的最小loss
    best_val_loss = 10000

    for epoch in range(params.epochs):
        # train
        train_loss = train_epoch(
            model=model, train_dataloader=train_dataloader,
            optimizer=optimizer, scheduler=scheduler,
            epoch=epoch, params=params
        )
        # validate
        # valid_loss = valid_epoch(
        #     model=model, valid_dataloader=valid_dataloader,
        #     epoch=epoch, params=params
        # )
        break



def main():
    params = ParameterConfig()
    tokenizer = BertTokenizerFast(params.vocab_path)

    if not os.path.exists(params.save_model_path):
        os.mkdir(params.save_model_path)

    if params.pretrained_model:
        # 加载预训练模型
        model = GPT2LMHeadModel.from_pretrained(params.pretrained_model)
    else:
        # 初始化模型
        model_config = GPT2Config.from_json_file(params.config_json)
        model = GPT2LMHeadModel(config=model_config)
    model = model.to(params.device)

    assert model.config.vocab_size == tokenizer.vocab_size

    # 计算模型参数数量
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print(f"模型的参数总量-->{num_parameters}")
    # 加载训练集和验证集
    train_dataloader, valid_dataloader = get_dataloader(params.train_path, params.valid_path)
    print(f'train_dataloader-->{len(train_dataloader)}')
    print(f'validate_dataloader-->{len(valid_dataloader)}')
    train(model, train_dataloader, valid_dataloader, params)



if __name__ == '__main__':
    main()
