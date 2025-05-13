import os
import time
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_scheduler
from utils.metric_utils import ClassEvaluator
from utils.common_utils import *
from data_handle.data_loader import *
from utils.verbalizer import Verbalizer
from pet_config import *
from tqdm import tqdm
from rich import print


pc = ProjectConfig()


def evaluate_model(model, metric, data_loader, tokenizer, verbalizer):
    model.eval()
    metric.reset()

    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            logits = model(input_ids=batch['input_ids'].to(pc.device),
                           token_type_ids=batch['token_type_ids'].to(pc.device),
                           attention_mask=batch['attention_mask'].to(pc.device)).logits
            # print("logits-->", logits.shape)
            mask_labels = batch["mask_labels"].numpy().tolist()

            for i in range(len(mask_labels)):  # 去掉label中的[PAD] token
                while tokenizer.pad_token_id in mask_labels[i]:
                    mask_labels[i].remove(tokenizer.pad_token_id)

            mask_labels = ["".join(tokenizer.convert_ids_to_tokens(t)) for t in mask_labels]
            # print("mask_labels-->", mask_labels)

            predictions = convert_logits_to_ids(logits, batch["mask_positions"]).cpu().numpy().tolist()
            # print("predicitions-->", predictions)

            predictions = verbalizer.batch_find_main_label(predictions)
            # print(f"找到模型预测的子标签对应的主标签的结果--》{predictions}')")

            predictions = [ele['label'] for ele in predictions]
            # print(f"只获得预测的主标签的结果string--》{predictions}')")

            metric.add_batch(pred_batch=predictions, gold_batch=mask_labels)

        eval_metric = metric.compute()
        model.train()
    return eval_metric['accuracy'], eval_metric['precision'], \
           eval_metric['recall'], eval_metric['f1'], \
           eval_metric['class_metrics']

def model2train():
    model = AutoModelForMaskedLM.from_pretrained(pc.pre_model)
    tokenizer = AutoTokenizer.from_pretrained(pc.pre_model)
    verbalizer = Verbalizer(verbalizer_file=pc.verbalizer,
                            tokenizer=tokenizer,
                            max_label_len=pc.max_label_len)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": pc.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=pc.learning_rate)
    model.to(pc.device)

    train_dataloader, dev_dataloader = get_data()
    # 根据训练轮数计算最大训练步数，以便于scheduler动态调整lr
    num_update_steps_per_epoch = len(train_dataloader)
    # 指定总的训练步数，它会被学习率调度器用来确定学习率的变化规律，确保学习率在整个训练过程中得以合理地调节
    max_train_steps = pc.epochs * num_update_steps_per_epoch
    warm_steps = int(pc.warmup_ratio*max_train_steps)  # 预热阶段的训练步数

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps
    )

    loss_list = []
    tic_train = time.time()
    metric = ClassEvaluator()
    criterion = torch.nn.CrossEntropyLoss()
    global_step, best_f1 = 0, 0
    print("开始训练")

    for epoch in range(pc.epochs):
        for batch in tqdm(train_dataloader):
            logits = model(
                input_ids=batch["input_ids"].to(pc.device),  # [2, 256]
                token_type_ids=batch["token_type_ids"].to(pc.device),  # [2, 256]
                attention_mask=batch["attention_mask"].to(pc.device)  # [2, 256]
            ).logits

            # 真实标签
            mask_labels = batch["mask_labels"].numpy().tolist()  # [[2354, 4456], [...]]
            sub_labels = verbalizer.batch_find_sub_labels(mask_labels)
            sub_labels = [ele["token_ids"] for ele in sub_labels]  # [[[2354, 4556], [3466, 5678]], [[...], [...]]]

            loss = mlm_loss(
                logits,
                batch["mask_positions"].to(pc.device),
                sub_labels,
                criterion,
                pc.device
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            loss_list.append(loss)
            global_step += 1

            # 打印训练日志
            if global_step % pc.logging_steps == 0:
                time_diff = time.time() - tic_train
                loss_avg = sum(loss_list) / len(loss_list)
                print("global step: %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                      % (global_step, epoch, loss_avg, pc.logging_steps / time_diff))
                tic_train = time.time()

            # 模型验证
            if global_step % pc.valid_steps == 0:
                cur_save_dir = os.path.join(pc.save_dir, "model_%d" % global_step)
                if not cur_save_dir:
                    os.makedirs(cur_save_dir)
                model.save_pretrained(cur_save_dir)
                tokenizer.save_pretrained(cur_save_dir)

                acc, precision, recall, f1, class_metrics = evaluate_model(model,
                                                                           metric,
                                                                           dev_dataloader,
                                                                           tokenizer,
                                                                           verbalizer)
                print("Evaluation precision: %.5f, recall: %.5f, F1: %.5f" % (precision, recall, f1))

                if f1 > best_f1:
                    print(
                        f"best F1 performence has been updated: {best_f1:.5f} --> {f1:.5f}"
                    )
                    print(f'Each Class Metrics are: {class_metrics}')
                    best_f1 = f1
                    cur_save_dir = os.path.join(pc.save_dir, "model_best")
                    if not os.path.exists(cur_save_dir):
                        os.makedirs(cur_save_dir)
                    model.save_pretrained(os.path.join(cur_save_dir))
                    tokenizer.save_pretrained(os.path.join(cur_save_dir))
    print("结束训练")


if __name__ == '__main__':
    model2train()