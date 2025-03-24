from fastapi import FastAPI, Query
from api import analyze_news

app = FastAPI()

@app.get("/analyze")
def analyze(company_name: str = Query(..., description="Enter the company name (e.g., Tesla, Apple)")):
    return analyze_news(company_name)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

