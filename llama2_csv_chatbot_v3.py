''' CSV Chatbot v3 - Testing the use of langchain llms, prompts and chains 

based on this walkthrough https://python.langchain.com/docs/integrations/llms/llamacpp

No UI

Run Instructions:
1. conda activate test_env (??)
2. install requirements?
3. run - python llama2_csv_chatbot_v3.py

'''

from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# model_file = "/Users/wandacosta/llama2_local/models/llama-2-7b-chat.ggmlv3.q4_K_M.bin"

template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure that we have the right answer."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="/Users/wandacosta/llama2_local/models/llama-2-7b-chat.ggmlv3.q4_K_M.bin",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager, 
    verbose=True, # Verbose is required to pass to the callback manager
)

prompt = """
Question: A rap battle between Stephen Colbert and John Oliver
"""
result = llm(prompt)
print(prompt, " ", result)