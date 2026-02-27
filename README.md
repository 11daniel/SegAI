# SegmentIQ

SegmentIQ is an E-Wallet Intelligence Streamlit application. It provides Customer Segmentation, Time-Series Forecasting, and acts as an AI Advisor via OpenAI GPT.

## Getting Started

### Prerequisites

- Python 3.8+
- [OpenAI API Key](https://platform.openai.com/) (Optional, for the AI Advisor feature)

### Installation

1. Create and activate a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Setup environment variables:
Create a `.env` file based on the example:
```bash
cp .env.example .env
```
Add your `OPENAI_API_KEY` to the `.env` file for the AI Advisor feature to work.

### Running the App

Run the application using Streamlit:
```bash
streamlit run app.py
```
Wait for the local server to start and then visit `http://localhost:8501` to view your application!
