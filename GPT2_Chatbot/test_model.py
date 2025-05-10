from transformers import GPT2LMHeadModel, GPT2Config
from parameter_config import *

params = ParameterConfig()

# 创建模型
if params.pretrained_model:
    model = GPT2LMHeadModel.from_pretrained(params.pretrained_model)
else:
    model_config = GPT2Config.from_json_file(params.config_json)
    model = GPT2LMHeadModel(config=model_config)

print(f"model->{model}")