from transformers import BertTokenizerFast
import pickle
from tqdm import tqdm


def data_preprocess(train_txt_path, train_pkl_path):
    tokenizer = BertTokenizerFast(r"C:\Users\19981\Documents\GitHub\LLM\GPT2_Chatbot\vocab\vocab.txt")
    sep_id = tokenizer.sep_token_id  # 获取分隔符[SEP]的token ID
    cls_id = tokenizer.cls_token_id  # 获取起始符[CLS]的token ID

    with open(train_txt_path, "r", encoding="utf-8") as f:
        data = f.read()
    train_data = data.split("\n\n")

    # 开始进行tokenize
    # 保存所有的对话数据,每条数据的格式为："[CLS]seq1[SEP]seq2[SEP]seq3[SEP]"
    dialogue_len = []  # 记录所有对话tokenize分词之后的长度，用于统计中位数与均值
    dialogue_list = []  # 记录所有对话
    for index, dialogue in enumerate(tqdm(train_data)):
        # 分开问题和答案
        sequences = dialogue.split("\n")
        input_ids = [cls_id]
        for sequence in sequences:
            input_ids += tokenizer.encode(sequence, add_special_tokens=False)
            input_ids.append(sep_id)
        dialogue_len.append(len(input_ids))  # 将对话的tokenize后的长度添加到对话长度列表中
        dialogue_list.append(input_ids)  # 将tokenize后的对话添加到对话列表中

    # 保存数据
    with open(train_pkl_path, "wb") as f:
        pickle.dump(dialogue_list, f)


if __name__ == '__main__':
    train_txt_path = "../data/medical_train.txt"
    train_pkl_path = "../data/medical_train.pkl"
    data_preprocess(train_txt_path, train_pkl_path)