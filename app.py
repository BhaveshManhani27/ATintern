import gradio as gr
from api import analyze_news

def analyze(company_name: str):
    # Run your analysis
    result = analyze_news(company_name)
    # Assume text_to_speech saves the audio to "output_speech.mp3"
    audio_path = "output_speech.mp3"
    # Return both JSON result and the audio file path.
    return result, audio_path

iface = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(lines=1, placeholder="Enter company name (e.g., Tesla, Apple)"),
    outputs=[ "json", gr.Audio(type="filepath") ],  # Specify that audio is a file path
    title="News Analyzer",
    description="Enter a company name to analyze its news and hear the Hindi audio summary."
)

if __name__ == "__main__":
    iface.launch()

