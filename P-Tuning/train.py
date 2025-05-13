import os
import time
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_scheduler
from tqdm import tqdm
from utils.metric_utils import *
from utils.common_utils import *
from data_handle.data_loader import *
from utils.verbalizer import *
from ptune_config import *
from transformers import Trainer

pc = ProjectConfig()

def model2train():
    model = AutoModelForMaskedLM.from_pretrained(pc.pre_model)
    print("model-->", model)


if __name__ == '__main__':
    model2train()