# AI Customer Support Ticket Triage

An AI-powered customer support ticket triage system that automatically analyzes support tickets, predicts their category and urgency, and routes them to the appropriate support queue.

## Features

- Customer support ticket classification
- Category prediction
- Urgency prediction
- Confidence scoring
- Human review for low-confidence predictions
- Automatic support queue routing
- Prediction history
- Analytics dashboard
- CSV export
- Streamlit web interface

## Supported Categories

- Billing
- Technical
- Account
- Product

## Supported Urgency Levels

- Low
- Medium
- High

## Queue Routing

| Category | Support Queue |
|---|---|
| Billing | Billing Support |
| Technical | Technical Support |
| Account | Account Support |
| Product | Product Support |

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Logistic Regression
- Joblib
- Streamlit
- Matplotlib
- Seaborn
- Git
- GitHub

## Project Structure

```text
AI-Customer-Support-Triage/
│
├── models/
│   ├── category_model.pkl
│   └── urgency_model.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore