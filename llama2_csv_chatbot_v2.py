''' CSV Chatbot v2 - Testing the use of tokenizers, vectorestore, and llama_index

based on 'Porsche 911 Data Analysis & Query with Llama2' on Kaggle
https://www.kaggle.com/code/wcosta/porsche-911-data-analysis-query-with-llama-2-7b/edit

No UI

Run Instructions:
1. conda activate test_env
2. install requirements?
3. Run - python llama2_csv_chatbot_v2.py

'''

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index import VectorStoreIndex, ServiceContext, set_global_service_context, Document
from llama_index.llms import HuggingFaceLLM
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


data_file = "/Users/wandacosta/llama2_local/titanic3.csv"

# load the csv data into a pandas dataframe
df = pd.read_csv(data_file)

# Convert the DataFrame content into a format suitable for Llama
documents = [
    Document(
        text=" ".join([f"{col}: {value}" for col, value in zip(df.columns, row.astype(str))]),
        metadata={"row_num": idx}
    ) 
    for idx, row in df.iterrows()
]

# print(documents)

# Llama setup
model_name = "/Users/wandacosta/llama2_local/models"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,device_map='auto',torch_dtype=torch.float16)

system_prompt = """<s>[INST] <<SYS>>
You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Your goal is to provide answers relating to the car Porsche from the csv.<</SYS>>"""

query_wrapper_prompt = "{query_str}"

llm = HuggingFaceLLM(
    context_window=4098,
    max_new_tokens=256,
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    model=model,
    tokenizer=tokenizer
)

embeddings = LangchainEmbedding(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
service_context = ServiceContext.from_defaults(chunk_size=4098, llm=llm, embed_model=embeddings)
set_global_service_context(service_context)

# Create an index using the DataFrame's content
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

def generate_response(query_text):
    input_tokens = tokenizer(query_text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    output = model.generate(**input_tokens)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Sample Queries
queries = [
    "Describe the data set.",
    "How many passengers survived on the titanic?"
]

for query in queries:
    print(f"Question: {query}")
    print(f"Response: {generate_response(query)}")
    print("-" * 50)