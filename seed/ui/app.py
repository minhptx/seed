import numpy as np
import gradio as gr
import pandas as pd

demo = gr.Blocks()

def update_df(file):
    df = pd.read_csv(file.name)
    print(df)
    return df

def detect_error(df):
    pass

with demo:
    gr.Markdown("Upload a csv files:")
    file = gr.File()
    
    dataframe = gr.DataFrame()
    file.change(fn=update_df, inputs=file, outputs=dataframe)
    submit = gr.Button("Detect errors")
    output = gr.HTML()
    submit.click(fn=detect_error, inputs=dataframe, outputs=output)


    
demo.launch(share=True)