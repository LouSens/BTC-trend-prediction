# BTC Trend Prediction Pipeline

## Overview
This project implements an 18-day methodology for predicting Bitcoin (BTC) daily price trends using Machine Learning.

## Methodology (18-Day Sprint)
- **Phase 1-3**: EDA & Data Discovery
- **Phase 4-7**: Preprocessing & Feature Engineering
- **Phase 8-12**: Model Training & Hyperparameter Tuning
- **Phase 13-15**: Evaluation & Backtesting
- **Phase 16-18**: UI Deployment with Streamlit

## Project Structure
- `data/`: Raw and processed datasets.
- `notebooks/`: Experimental Jupyter notebooks.
- `src/`: Reusable Python modules and the main pipeline script.
- `app/`: Streamlit web interface for predictions.

## Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Download the Kaggle BTC-USD dataset and place it in `data/raw/`.
4. Run the notebooks or the pipeline script.
5. Launch the app: `streamlit run app/app.py`