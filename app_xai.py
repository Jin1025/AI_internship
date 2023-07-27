# Import packages
import pandas as pd
import gradio as gr
import pickle
import matplotlib.pyplot as plt

import sys; sys.path.append("dist")

from dist.magic_canai import (
    load_state,
    inverse_transform,
    model_with_transformers as model,
    get_local_explanation,
    gpt_interpretation,
    find_counterfactuals,
)

# Set OPENAI_API_KEY for gpt_interpretation
import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the model state file
state = load_state(f'병충해 예측 변수 데이터 - sheet4 100세로_변수5개_state.pkl')
tests = inverse_transform(state, state.X_test)

point, prediction, importances = None, None, None

# The workhorses
def predict(*args):
    global point, prediction, importances
    point = pd.DataFrame({
        'pH_level': [args[0]],
        'water_given_per_week': [args[1]],
        'temperature': [args[2]],
        'wind': [args[3]],
        'month': [args[4]]
    })

    prediction = str(model(state).predict(point)[0])
    importances = pd.DataFrame(
        get_local_explanation(state, point), columns=["Feature", "Importance"]
    )
    return f'**disease_insect: {prediction}**', importances

def explain(*args):
    if point is None or prediction is None or importances is None:
        explanation = "Make prediction first!"
    else:
        explanation = gpt_interpretation(state, point, prediction, importances) 
    return explanation

def counterfactuals(*args):
    if point is None or prediction is None or importances is None:
        # counterfactuals = "Make prediction first!"
        counterfactuals = pd.DataFrame()
    else:
        counterfactuals = find_counterfactuals(state, point, tests).iloc[:50]
    return counterfactuals

# User Interface
number = gr.components.Number
radio = gr.components.Radio
dropdown = gr.components.Dropdown
text = gr.components.Textbox

with gr.Blocks() as service:
    gr.Markdown(value="<h4>병충해 예측 변수 데이터 - sheet4 100세로 변수5개 'disease_insect' Predictor and Explainer</h4>Powered by <b>Magic canAI</b>")

    # User inputs
    pHlevel = number(value=4.0, label="pH_level")
    watergivenperweek = number(value=6, label="water_given_per_week")
    temperature = number(value=26, label="temperature")
    wind = number(value=1.1, label="wind")
    month = number(value=7, label="month")

    # Controls
    predict_btn = gr.components.Button("Predict")
    predict_outputs = [
        gr.Markdown(),
        gr.DataFrame(label="Feature Importances"),
    ]
    predict_btn.click(
        fn=predict,
        inputs=[pHlevel, watergivenperweek, temperature, wind, month],
        outputs=predict_outputs
    )
    explain_btn = gr.components.Button("Explain Prediction")
    explain_btn.click(
        fn=explain,
        inputs=[],
        outputs=text(show_label=False),
    )
    counter_btn = gr.components.Button("Counterfactuals")
    counter_btn.click(
        fn=counterfactuals,
        inputs=[],
        outputs=gr.DataFrame(),
    )

    # Launch the app
    service.launch(debug=True)
###
