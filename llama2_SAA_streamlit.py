'''Created by Wanda Costa on Oct 20, 2023

run instructions:
conda activate Llama2_env
python llama2_SAA_streamlit.py
streamlit run llama2_SAA_streamlit.py

'''
# import fire
# import torch
import pandas as pd
from llama_cpp import Llama
import streamlit as st
from llama_chat_format import format_to_llama_chat_style
from langchain import HuggingFacePipeline
# from langchain.formatting import format_csv
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.llms import LlamaCpp
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

# model_name="TheBloke/Llama-2-7B-Chat-GGML"
file_name="/Users/wandacosta/llama2_local/models/llama-2-7b-chat.ggmlv3.q4_K_M.bin"
history = []
# model = Llama(file_name, n_ctx=4096)
llm = LlamaCpp(model_path=file_name, temperature=0.6, max_tokens=512, top_p=0.9)
# tokenizer = None #GGML models don't require tokenizer, but GPTQ models will

# def convert_csv_to_prompt(df):
#     formatted_csv = format_csv(df)
#     return f"Here is the formatted data: {formatted_csv}"

# Define a function to interact with the chatbot
def generate_response(model, history, df):
    # print("history = ", history)
    user_message = history[-1][0]
    # csv_prompt = convert_csv_to_prompt(df)
    prompt = f"{user_message}\n{df}"
    # instruction = format_to_llama_chat_style(prompt)
    # print(instruction)
    # kwargs = dict(temperature=0.6, top_p=0.9)
    # kwargs["max_tokens"] = 512
    full_response = model(prompt)
    # print("full_response = ", full_response)
    token = full_response["choices"][0]["text"]
    # print("token = ", token)
    return token

st.title("Llama2 Chatbot")
st.write('Prompt-driven data analytics and visualization code generation with Llama2')

# Add widget to upload file for analysis.
uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=["csv"])

if uploaded_file is not None:
    # df = pd.read_csv(uploaded_file)
    loader = CSVLoader(uploaded_file)
    # file_contents = uploaded_file.read()
    df = loader.load()
    print(df)
    st.write(df.head(3))
    # data_col = list(df.columns)

user_message = st.text_area("Enter your prompt:")
prompt = user_message

if st.button("Generate"):
    if prompt:
        with st.spinner("Generating response..."):
            history.append([prompt, ""])
            response = generate_response(llm, history, df)
            st.write(response)
            history[-1][1] += response
    else:
        st.warning("Please enter a prompt.")
            


