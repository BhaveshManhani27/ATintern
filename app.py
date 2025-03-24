# from fastapi import FastAPI, Query
# from api import analyze_news

# app = FastAPI()

# @app.get("/analyze")
# def analyze(company_name: str = Query(..., description="Enter the company name (e.g., Tesla, Apple)")):
#     return analyze_news(company_name)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

import gradio as gr
from api import analyze_news

def analyze(company_name: str):
    # Call your existing function that does the analysis.
    return analyze_news(company_name)

# Create the Gradio interface.
iface = gr.Interface(
    fn=analyze,
    inputs=gr.components.Textbox(lines=1, placeholder="Enter the company name (e.g., Tesla, Apple)"),
    outputs="json",
    title="News Analyzer",
    description="Enter a company name to analyze its news."
)

# Launch the Gradio app.
if __name__ == "_main_":
    iface.launch()

