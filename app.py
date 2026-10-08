import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Customer Support Ticket Triage",
    page_icon="🎫",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

CATEGORY_MODEL_PATH = "models/category_model.pkl"
URGENCY_MODEL_PATH = "models/urgency_model.pkl"

# IMPORTANT:
# Prediction history is stored directly in the project folder.
# No "logs" folder is required.
HISTORY_FILE = "prediction_history.csv"


# ============================================================
# CONSTANTS
# ============================================================

QUEUE_MAPPING = {
    "billing": "Billing Support",
    "technical": "Technical Support",
    "account": "Account Support",
    "product": "Product Support"
}

HISTORY_COLUMNS = [
    "Ticket",
    "Category",
    "Category Confidence",
    "Urgency",
    "Urgency Confidence",
    "Queue",
    "Routing Status",
    "Human Review"
]


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Cleans ticket text before sending it to the ML models.
    """

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # Remove special characters
    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    try:

        category_model = joblib.load(
            CATEGORY_MODEL_PATH
        )

        urgency_model = joblib.load(
            URGENCY_MODEL_PATH
        )

        return category_model, urgency_model

    except FileNotFoundError as e:

        st.error(
            "❌ Model file not found."
        )

        st.info(
            "Make sure your project has this structure:\n\n"
            "AI-Customer-Support-Triage/\n"
            "├── app.py\n"
            "├── models/\n"
            "│   ├── category_model.pkl\n"
            "│   └── urgency_model.pkl\n"
            "└── prediction_history.csv"
        )

        st.error(
            f"Missing file: {e}"
        )

        st.stop()

    except Exception as e:

        st.error(
            f"❌ Error loading models: {e}"
        )

        st.stop()


category_model, urgency_model = load_models()


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def predict_category(ticket):

    cleaned_ticket = clean_text(ticket)

    prediction = category_model.predict(
        [cleaned_ticket]
    )[0]

    probabilities = category_model.predict_proba(
        [cleaned_ticket]
    )[0]

    confidence = float(
        np.max(probabilities)
    )

    return prediction, confidence


def predict_urgency(ticket):

    cleaned_ticket = clean_text(ticket)

    prediction = urgency_model.predict(
        [cleaned_ticket]
    )[0]

    probabilities = urgency_model.predict_proba(
        [cleaned_ticket]
    )[0]

    confidence = float(
        np.max(probabilities)
    )

    return prediction, confidence


# ============================================================
# TRIAGE FUNCTION
# ============================================================

def triage_ticket(
    ticket,
    confidence_threshold
):

    category, category_confidence = predict_category(
        ticket
    )

    urgency, urgency_confidence = predict_urgency(
        ticket
    )

    queue = QUEUE_MAPPING.get(
        category,
        "General Support"
    )

    # Check confidence
    if (
        category_confidence >= confidence_threshold
        and
        urgency_confidence >= confidence_threshold
    ):

        routing_status = "Automatically Routed"
        human_review = "No"

    else:

        routing_status = "Human Review Required"
        human_review = "Yes"

    return {
        "category": category,
        "category_confidence": category_confidence,
        "urgency": urgency,
        "urgency_confidence": urgency_confidence,
        "queue": queue,
        "routing_status": routing_status,
        "human_review": human_review
    }


# ============================================================
# LOAD PREDICTION HISTORY
# ============================================================

def load_prediction_history():

    if os.path.exists(HISTORY_FILE):

        try:

            history_df = pd.read_csv(
                HISTORY_FILE
            )

            return history_df

        except Exception:

            return pd.DataFrame(
                columns=HISTORY_COLUMNS
            )

    return pd.DataFrame(
        columns=HISTORY_COLUMNS
    )


# ============================================================
# SAVE PREDICTION HISTORY
# ============================================================

def save_prediction_history(history_df):

    # IMPORTANT:
    # No os.makedirs("logs") here.
    # This prevents the WinError 183 problem.

    history_df.to_csv(
        HISTORY_FILE,
        index=False
    )


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title(
    "🎫 AI Customer Support Ticket Triage"
)

st.markdown(
    """
    ### 🤖 Intelligent Support Ticket Classification

    This application automatically analyzes customer support tickets
    and predicts:

    - 🏷️ **Ticket Category**
    - 🚨 **Urgency Level**
    - 🎯 **Prediction Confidence**
    - 📥 **Support Queue**
    - 👨‍💼 **Human Review Requirement**

    The system uses **TF-IDF + Logistic Regression** machine learning models.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "⚙️ Triage Settings"
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.50,
    max_value=0.95,
    value=0.70,
    step=0.05
)

st.sidebar.write(
    f"Current threshold: **{confidence_threshold:.0%}**"
)

st.sidebar.info(
    """
    If either category or urgency confidence
    falls below this threshold, the ticket
    will be sent for human review.
    """
)


# ============================================================
# TICKET INPUT
# ============================================================

st.header(
    "📝 Analyze Support Ticket"
)

ticket_text = st.text_area(
    "Enter customer support ticket:",
    height=180,
    placeholder=(
        "Example:\n"
        "I was charged twice for my subscription "
        "and I need a refund immediately."
    )
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze_button = st.button(
    "🔍 Analyze Ticket",
    type="primary",
    use_container_width=True
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze_button:

    if not ticket_text.strip():

        st.warning(
            "⚠️ Please enter a support ticket first."
        )

    else:

        with st.spinner(
            "Analyzing ticket..."
        ):

            result = triage_ticket(
                ticket_text,
                confidence_threshold
            )

        # ----------------------------------------------------
        # RESULT HEADER
        # ----------------------------------------------------

        st.success(
            "✅ Ticket analysis completed!"
        )

        st.divider()

        st.subheader(
            "📊 Prediction Results"
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Category",
                result["category"].title()
            )

        with col2:

            st.metric(
                "Urgency",
                result["urgency"].title()
            )

        with col3:

            st.metric(
                "Support Queue",
                result["queue"]
            )

        with col4:

            st.metric(
                "Routing",
                result["routing_status"]
            )


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        st.subheader(
            "🎯 Prediction Confidence"
        )

        confidence_col1, confidence_col2 = st.columns(2)

        with confidence_col1:

            st.write(
                f"**Category Confidence: "
                f"{result['category_confidence']:.2%}**"
            )

            st.progress(
                result["category_confidence"]
            )

        with confidence_col2:

            st.write(
                f"**Urgency Confidence: "
                f"{result['urgency_confidence']:.2%}**"
            )

            st.progress(
                result["urgency_confidence"]
            )


        # ----------------------------------------------------
        # ROUTING DECISION
        # ----------------------------------------------------

        st.subheader(
            "📥 Routing Decision"
        )

        if result["routing_status"] == "Automatically Routed":

            st.success(
                f"✅ **Automatically Routed** to "
                f"**{result['queue']}**"
            )

        else:

            st.warning(
                "⚠️ **Human Review Required**"
            )

            st.write(
                "The model confidence is below "
                "the configured threshold."
            )


        # ----------------------------------------------------
        # HUMAN REVIEW
        # ----------------------------------------------------

        if result["human_review"] == "Yes":

            st.error(
                "👨‍💼 This ticket should be reviewed by a human agent."
            )

        else:

            st.success(
                "🤖 This ticket can be automatically routed."
            )


        # ----------------------------------------------------
        # DETAILED PREDICTION
        # ----------------------------------------------------

        with st.expander(
            "🔎 View Detailed Prediction"
        ):

            prediction_data = pd.DataFrame(
                {
                    "Field": [
                        "Ticket",
                        "Category",
                        "Category Confidence",
                        "Urgency",
                        "Urgency Confidence",
                        "Support Queue",
                        "Routing Status",
                        "Human Review"
                    ],

                    "Value": [
                        ticket_text,
                        result["category"].title(),
                        f"{result['category_confidence']:.2%}",
                        result["urgency"].title(),
                        f"{result['urgency_confidence']:.2%}",
                        result["queue"],
                        result["routing_status"],
                        result["human_review"]
                    ]
                }
            )

            st.dataframe(
                prediction_data,
                use_container_width=True,
                hide_index=True
            )


        # ----------------------------------------------------
        # SAVE PREDICTION
        # ----------------------------------------------------

        history_df = load_prediction_history()

        new_prediction = pd.DataFrame(
            [
                {
                    "Ticket": ticket_text,
                    "Category": result["category"],
                    "Category Confidence":
                        round(
                            result["category_confidence"],
                            4
                        ),
                    "Urgency": result["urgency"],
                    "Urgency Confidence":
                        round(
                            result["urgency_confidence"],
                            4
                        ),
                    "Queue": result["queue"],
                    "Routing Status":
                        result["routing_status"],
                    "Human Review":
                        result["human_review"]
                }
            ]
        )

        history_df = pd.concat(
            [
                history_df,
                new_prediction
            ],
            ignore_index=True
        )

        save_prediction_history(
            history_df
        )

        st.info(
            "💾 Prediction saved to prediction_history.csv"
        )


# ============================================================
# ANALYTICS DASHBOARD
# ============================================================

st.divider()

st.header(
    "📈 Analytics Dashboard"
)

history_df = load_prediction_history()


# ============================================================
# NO HISTORY
# ============================================================

if history_df.empty:

    st.info(
        "📭 No prediction history available yet. "
        "Analyze some tickets to see analytics."
    )


# ============================================================
# DASHBOARD
# ============================================================

else:

    # --------------------------------------------------------
    # SUMMARY METRICS
    # --------------------------------------------------------

    total_tickets = len(
        history_df
    )

    automatically_routed = len(
        history_df[
            history_df["Routing Status"]
            == "Automatically Routed"
        ]
    )

    human_review_count = len(
        history_df[
            history_df["Human Review"]
            == "Yes"
        ]
    )

    avg_category_confidence = (
        history_df["Category Confidence"]
        .mean()
    )

    avg_urgency_confidence = (
        history_df["Urgency Confidence"]
        .mean()
    )


    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:

        st.metric(
            "Total Tickets",
            total_tickets
        )

    with metric2:

        st.metric(
            "Auto Routed",
            automatically_routed
        )

    with metric3:

        st.metric(
            "Human Review",
            human_review_count
        )

    with metric4:

        st.metric(
            "Avg Confidence",
            f"{avg_category_confidence:.1%}"
        )


    # --------------------------------------------------------
    # CATEGORY DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "🏷️ Ticket Category Distribution"
    )

    category_counts = (
        history_df["Category"]
        .value_counts()
    )

    st.bar_chart(
        category_counts,
        height=250
    )


    # --------------------------------------------------------
    # URGENCY DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "🚨 Ticket Urgency Distribution"
    )

    urgency_counts = (
        history_df["Urgency"]
        .value_counts()
    )

    st.bar_chart(
        urgency_counts,
        height=250
    )


    # --------------------------------------------------------
    # QUEUE DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "📥 Support Queue Distribution"
    )

    queue_counts = (
        history_df["Queue"]
        .value_counts()
    )

    st.bar_chart(
        queue_counts,
        height=250
    )


    # --------------------------------------------------------
    # CONFIDENCE SUMMARY
    # --------------------------------------------------------

    st.subheader(
        "🎯 Confidence Summary"
    )

    confidence_summary = pd.DataFrame(
        {
            "Confidence": [
                avg_category_confidence,
                avg_urgency_confidence
            ]
        },
        index=[
            "Category",
            "Urgency"
        ]
    )

    st.bar_chart(
        confidence_summary,
        height=250
    )


    # --------------------------------------------------------
    # PREDICTION HISTORY
    # --------------------------------------------------------

    st.subheader(
        "📋 Prediction History"
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # DOWNLOAD CSV
    # --------------------------------------------------------

    st.subheader(
        "⬇️ Export Data"
    )

    csv_data = history_df.to_csv(
        index=False
    )

    st.download_button(
        label="📥 Download Prediction History",
        data=csv_data,
        file_name="prediction_history.csv",
        mime="text/csv",
        use_container_width=True
    )


    # --------------------------------------------------------
    # CLEAR HISTORY
    # --------------------------------------------------------

    st.subheader(
        "🗑️ Clear Prediction History"
    )

    if st.button(
        "Clear All History",
        type="secondary"
    ):

        if os.path.exists(
            HISTORY_FILE
        ):

            try:

                os.remove(
                    HISTORY_FILE
                )

                st.success(
                    "✅ Prediction history cleared."
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"❌ Could not clear history: {e}"
                )

        else:

            st.info(
                "No prediction history file exists."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Customer Support Ticket Triage | "
    "TF-IDF + Logistic Regression + Streamlit"
)