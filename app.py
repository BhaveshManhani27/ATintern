import gradio as gr
from api import analyze_news

def analyze(company_name: str):
    try:
        # Directly call the analysis function.
        result = analyze_news(company_name)
    except Exception as e:
        result = {"error": str(e)}
    return result

iface = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(lines=1, placeholder="Enter company name (e.g., Tesla, Apple)"),
    outputs="json",
    title="News Analyzer",
    description="Enter a company name to analyze its news."
)

if __name__ == "__main__":
    iface.launch()

