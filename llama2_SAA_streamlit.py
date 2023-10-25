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
from streamlit_chat import message
import tempfile
from llama_chat_format import format_to_llama_chat_style
from langchain import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
# from langchain.formatting import format_csv
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.llms import LlamaCpp, CTransformers
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

DB_FAISS_PATH = "vectorstore/db_faiss"

# model_name="TheBloke/Llama-2-7B-Chat-GGML"
file_name="/Users/wandacosta/llama2_local/models/llama-2-7b-chat.ggmlv3.q4_K_M.bin"
# history = []
# model = Llama(file_name, n_ctx=4096)
# loading the model
def load_llm(file_name):
    # load the locally downloaded model here
    llm = CTransformers(
        model = file_name,
        model_type="llama",
        max_new_tokens=512,
        temperature=0.6
    )
    return llm
# llm = LlamaCpp(model_path=file_name, temperature=0.6, max_tokens=512, top_p=0.9)
# tokenizer = None #GGML models don't require tokenizer, but GPTQ models will

# def convert_csv_to_prompt(df):
#     formatted_csv = format_csv(df)
#     return f"Here is the formatted data: {formatted_csv}"

# def prompt_format(user_message):
#     B_INST, E_INST = "[INST]", "[/INST]"
#     B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
#     DEFAULT_SYSTEM_PROMPT = """\
#     You are an expert data analytics assistant. Please ensure that your responses are accurate and detailed.

#     If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct."""

#     prompt = ""
#     instruction = f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}" 
#     request = f"{B_INST} {user_message} {E_INST}"
#     prompt = instruction + request
#     print("prompt = ", prompt)
#     return prompt

# Define a function to interact with the chatbot
def generate_response(prompt):
    result = chain({"question": prompt, "chat_history": st.session_state['history']})
    st.session_state['history'].append((prompt, result["answer"]))
    # print("history = ", history)
    # user_message = history[-1][0]
    # csv_prompt = convert_csv_to_prompt(df)
    # request = f"{prompt}\n{df}"
    # instruction = format_to_llama_chat_style(prompt)
    # print(instruction)
    # kwargs = dict(temperature=0.6, top_p=0.9)
    # kwargs["max_tokens"] = 512
    # full_response = model(request)
    # print("full_response = ", full_response)
    # token = full_response["choices"][0]["text"]
    # print("token = ", token)
    return result["answer"]

st.title("Llama2 Chatbot")
st.write('Prompt-driven data analytics and visualization with Llama2 🦙 and LangChain 🦜')

# Add widget to upload file for analysis.
uploaded_file = st.sidebar.file_uploader("Upload a CSV file for analysis", type=["csv"])
print("uploaded_file = ", uploaded_file)
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
    # df = pd.read_csv(uploaded_file)
    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    # file_contents = uploaded_file.read()
    df = loader.load()
    # print(df)
    st.write(df)
    # data_col = list(df.columns)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    db = FAISS.from_documents(df, embeddings)
    db.save_local(DB_FAISS_PATH)
    llm = load_llm(file_name)
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())
    


if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello, Ask me anything about ", uploaded_file]
    
if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey!"]

#container for chat history
response_containter = st.container()

container = st.container()

with container:
    with st.form(key="my_form", clear_on_submit=True):
        user_prompt = st.text_input("Enter your prompt:")
        submit_button = st.form_submit_button(label='Submit')
# prompt = prompt_format(user_message)

    if submit_button and user_prompt:
        output = generate_response(user_prompt)
        st.session_state['past'].append(user_prompt)
        st.session_state['generated'].append(output)
    
if st.session_state['generated']:
    with response_containter:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i)+'_user', avatar_style="big_smile")
            message(st.session_state['generated'][i], key=str(i), avatar_style="square")
            
        # if prompt:
        #     with st.spinner("Generating response..."):
        #         # history.append([prompt, ""])
        #         response = generate_response(llm, prompt, df)
        #         st.write(response)
        #         # history[-1][1] += response
        # else:
        #     st.warning("Please enter a prompt.")
            


