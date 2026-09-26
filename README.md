# Email Threat Classification

This project aims to classify emails to detect potential threats using a machine learning model.

## Directory Structure

- `app.py`: The main application script (e.g., a Flask or FastAPI backend).
- `model/`: Directory containing the pre-trained machine learning model artifacts.
  - `model.pkl`: The trained classifier model.
  - `preprocessor.pkl`: The data preprocessor used for feature extraction before model inference.
- `requirements.txt`: Python package dependencies required to run the project.

## Setup and Installation

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

To start the application, run the following command:
```bash
streamlit run app.py
```
