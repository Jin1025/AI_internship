# -*- coding: cp949 -*-

import sys; sys.path.append("dist")

# Use this way
from dist import ffchat
# ...or this way
from dist.ffchat import (
    set_embeddings,
    download_embeddings,
    remove_embeddings,
    load_mydata_from_embeddings,
    load_mydata_from_source,
)

# Set OPENAI_API_KEY
import os
import openai
import gradio as gr

openai.api_key = "OPENAI_API_KEY"

# Test queries
def test(query):
    informed = '*' if ffchat.is_informed() else ''
    print(f"[{query}]{informed}\n")
    try:
        print(ffchat.ask(query))
    except Exception as e:
        print(f"*** {str(e)} ***")
    print("\n")

# Function to interact with the model and get the response
def get_model_response(query):
    informed = '*' if ffchat.is_informed() else ''
    try:
        return ffchat.ask(query)
    except Exception as e:
        return f"*** {str(e)} ***"

# Load embeddings and set to the model
def load_and_set_embeddings():
    set_embeddings(load_mydata_from_embeddings("embeddings/토마토 병만 있는embeddings.csv"))
    print(f"적용된 임베딩: {ffchat.embedding_names()}\n\n")

# Level setting
ffchat.set_model('gpt-3.5 (long)')      # 'gpt-3.5 (short)', 'gpt-4'
print(f"Model: {ffchat.get_model()}\n")
ffchat.set_creativity(1)
ffchat.set_expertise("Truth-teller")

# Gradio Interface
iface = gr.Interface(
    fn=get_model_response,
    inputs=gr.inputs.Textbox(),
    outputs=gr.outputs.Textbox(),
    live=True,
    capture_session=True  # Capture OpenAI API session to avoid repeated requests
)


# Load embeddings and set to the model
load_and_set_embeddings()


iface.launch(share=True)
