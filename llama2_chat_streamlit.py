'''Created by Wanda Costa on Oct 19, 2023

conda activate Llama2_env
python llama2_chat_streamlit.py

streamlit run llama2_chat_streamlit.py

'''
import fire
import torch
from llama_cpp import Llama
import streamlit as st
from llama_chat_format import format_to_llama_chat_style
from langchain import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

model_name="TheBloke/Llama-2-7B-Chat-GGML"
file_name="/Users/wandacosta/llama2_local/models/llama-2-7b-chat.ggmlv3.q4_K_M.bin"

model = Llama(file_name, n_ctx=4096)
tokenizer = None #GGML models don't require tokenizer, but GPTQ models will

# Define a function to interact with the chatbot
def generate_response(model, history):
    # print("history = ", history)
    instruction = format_to_llama_chat_style(history)
    # print(instruction)
    kwargs = dict(temperature=0.6, top_p=0.9)
    kwargs["max_tokens"] = 512
    full_response = model(prompt=instruction, **kwargs)
    # print("full_response = ", full_response)
    token = full_response["choices"][0]["text"]
    # print("token = ", token)
    return token

st.title("Llama2 Chatbot")

if "history" not in st.session_state.keys(): # Initialize the chat history
    st.session_state.history = []

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question."}
    ]   
    
if st.button("Clear"):  # Add a "Clear" button
    st.session_state.history = []  # Clear the conversation history
    st.session_state.messages = [{"role": "assistant", "content": "Ask me a question."}]  # Reset the chat messages

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # print(prompt)
    # if prompt:
    with st.spinner("Thinking..."):
        st.session_state.history.append([prompt, ""])
        response = generate_response(model, st.session_state.history)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.history[-1][1] += response
        st.write(response)

# print("messages =", st.session_state.messages)
# print("history = ", st.session_state.history)

