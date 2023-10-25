'''from kaggle

conda activate test_env
python kaggle_llama2.py

'''
import os
from transformers import AutoTokenizer
import transformers
import torch
import pandas as pd
import numpy as np

# model = "/Users/wandacosta/llama2_local/models/llama-2-7b-chat.ggmlv3.q4_K_M.bin"
# model = "/Users/wandacosta/llama2_local/models/"
model = "/kaggle/input/llama-2/pytorch/7b-chat-hf/1"

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline(
    'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")