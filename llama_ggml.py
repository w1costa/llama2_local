''' trying to clean up Thissenrand llama.py file to just include ggml chat version

run instructions:

conda activate Llama2_env
(?) pip install -r requirements.txt

python llama_ggml.py 

'''

# import os
import gradio as gr
import fire
from enum import Enum
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from transformers import TextIteratorStreamer
from llama_chat_format import format_to_llama_chat_style

def get_model_and_tokenizer(model_name, file_name):
    model = Llama(file_name, n_ctx=4096)
    tokenizer = None #GGML models don't require tokenizer, but GPTQ models will
    return model, tokenizer

def run_ui(model, tokenizer):
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.Button("Clear")

        def user(user_message, history):
            return "", history + [[user_message, None]]

        def bot(history):
            instruction = format_to_llama_chat_style(history)
            
            history[-1][1] = ""
            kwargs = dict(temperature=0.6, top_p=0.9)
            kwargs["max_tokens"] = 512
            for chunk in model(prompt=instruction, stream=True, **kwargs):
                token = chunk["choices"][0]["text"]
                history[-1][1] += token
                yield history

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
        clear.click(lambda: None, None, chatbot, queue=False)
        
        demo.queue()
        demo.launch(share=True, debug=True)

def main(model_name, file_name):
    assert model_name is not None, "model_name argument is missing."
    
    model, tokenizer = get_model_and_tokenizer(model_name, file_name)
    run_ui(model, tokenizer)

model_name="TheBloke/Llama-2-7B-Chat-GGML"
# file_name="llama-2-7b-chat.ggmlv3.q4_K_M.bin"
file_name="/Users/wandacosta/llama2_local/models/llama-2-7b-chat.ggmlv3.q4_K_M.bin"

if __name__ == '__main__':
    fire.Fire(main(model_name, file_name))