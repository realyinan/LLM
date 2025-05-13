import time
from typing import List
import torch
from rich import print
from transformers import AutoTokenizer, AutoModelForMaskedLM
from utils.verbalizer import Verbalizer
from data_handle.template import HardTemplate
from data_handle.data_preprocess import convert_example
from utils.common_utils import convert_logits_to_ids
from pet_config import *

pc = ProjectConfig()
device = pc.device
model_path = r'C:\Users\19981\Documents\GitHub\LLM\PET\checkpoints\model_best'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path)
model.to(device).eval()

max_label_len = 2
verbalizer = Verbalizer(
        verbalizer_file='data/verbalizer.txt',
        tokenizer=tokenizer,
        max_label_len=max_label_len
    )
prompt = open('./data/prompt.txt', 'r', encoding='utf8').readlines()[0].strip()
hard_template = HardTemplate(prompt=prompt)


def inference(contents: List[str]):
    with torch.no_grad():
        strat_time = time.time()
        examples = {
            "text": contents
        }
        tokenized_output = convert_example(
            examples,
            tokenizer,
            hard_template=hard_template,
            max_seq_len=pc.max_seq_len,
            max_label_len=pc.max_label_len,
            train_mode=False,
            return_tensor=True
        )
        logits = model(input_ids=tokenized_output['input_ids'].to(device),
                        token_type_ids=tokenized_output['token_type_ids'].to(device),
                        attention_mask=tokenized_output['attention_mask'].to(device)).logits
        predicitions = convert_logits_to_ids(logits, tokenized_output["mask_positions"]).cpu().numpy().tolist()
        predicitions = verbalizer.batch_find_main_label(predicitions)
        predicitions = [ele["label"] for ele in predicitions]
        return predicitions

if __name__ == '__main__':
    contents = [
        '天台很好看，躺在躺椅上很悠闲，因为活动所以我觉得性价比还不错，适合一家出行，特别是去迪士尼也蛮近的，下次有机会肯定还会再来的，值得推荐',
        '环境，设施，很棒，周边配套设施齐全，前台小姐姐超级漂亮！酒店很赞，早餐不错，服务态度很好，前台美眉很漂亮。性价比超高的一家酒店。强烈推荐',
        "物流超快，隔天就到了，还没用，屯着出游的时候用的，听方便的，占地小",
        "福行市来到无早集市，因为是喜欢的面包店，所以跑来集市看看。第一眼就看到了，之前在微店买了小刘，这次买了老刘，还有一直喜欢的巧克力磅蛋糕。好奇老板为啥不做柠檬磅蛋糕了，微店一直都是买不到的状态。因为不爱碱水硬欧之类的，所以期待老板多来点其他小点，饼干一直也是大爱，那天好像也没看到",
        "服务很用心，房型也很舒服，小朋友很喜欢，下次去嘉定还会再选择。床铺柔软舒适，晚上休息很安逸，隔音效果不错赞，下次还会来"
    ]
    print("针对下面的文本评论，请分别给出对应所属类别：")
    res = inference(contents)
    print('inference label(s):', res)
    new_dict = {}
    for i in range(len(contents)):
        new_dict[contents[i]] = res[i]
    print(new_dict)

