# ==========================================
# Online Course Completion Prediction System
# ==========================================

import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------
# Page Configuration
# ------------------------------------------

st.set_page_config(
    page_title="Online Course Completion Prediction",
    page_icon="🎓",
    layout="wide"
)

# ------------------------------------------
# Load Dataset
# ------------------------------------------

df = pd.read_csv("data/online_learning_course_completion_final.csv")

# ------------------------------------------
# Load ML Model & Encoders
# ------------------------------------------

model = joblib.load("models/course_completion_model.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

# ------------------------------------------
# Sidebar
# ------------------------------------------

st.sidebar.title("🎓 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "📊 Analysis",
        "🤖 Batch Prediction"
    ]
)


# ==========================================
# HOME PAGE
# ==========================================

if page == "🏠 Home":

    st.title("🎓 AI-Based Online Course Completion Prediction")

    st.markdown("""
Welcome to the **Online Course Completion Prediction System**.

This project analyzes learner behaviour and predicts the
completion status of online learners using Machine Learning.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Learners", len(df))
    col2.metric("ML Model", "Random Forest")
    col3.metric("Accuracy", "91%")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.subheader("Project Objective")

    st.info("""
Predict the completion status of an entire batch of learners
to help online learning platforms identify completed,
in-progress and dropped learners.
""")
    
# ==========================================
# ANALYSIS PAGE
# ==========================================

elif page == "📊 Analysis":

    st.title("📊 Exploratory Data Analysis")

    st.write("The following visualizations provide insights into learner behaviour and course completion patterns.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Completion Status")
        st.image("screenshots/Completion_status.JPG", use_container_width=True)

    with col2:
        st.subheader("Enrollment Month")
        st.image("screenshots/EnrolmentMonth_Dist.JPG", use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Completion Percentage")
        st.image("screenshots/Completion_percentage.JPG", use_container_width=True)

    with col4:
        st.subheader("Experience vs Completion")
        st.image("screenshots/EngatementScoreVSCompleStat.JPG", use_container_width=True)

    st.subheader("Correlation Heatmap")

    st.image("screenshots/Heatmap.JPG", use_container_width=True)



# ==========================================
# BATCH PREDICTION PAGE
# ==========================================

elif page == "🤖 Batch Prediction":

    st.title("🤖 Batch Prediction")

    uploaded_file = st.file_uploader(
        "Upload Student Batch CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        # ----------------------------
        # Read uploaded file
        # ----------------------------
        new_data = pd.read_csv(uploaded_file)

        # Keep original copy for displaying results
        result = new_data.copy()

        st.subheader("Uploaded Dataset")
        st.dataframe(result.head())

        # ----------------------------
        # Feature Engineering
        # ----------------------------

        new_data["Enrollment_Date"] = pd.to_datetime(
            new_data["Enrollment_Date"]
        )

        new_data["Enrollment_Month"] = (
            new_data["Enrollment_Date"].dt.month
        )

        new_data["Enrollment_Day"] = (
            new_data["Enrollment_Date"].dt.day
        )

        new_data.drop(
            columns=["Enrollment_Date"],
            inplace=True
        )

        # ----------------------------
        # Remove unnecessary columns
        # ----------------------------

        drop_columns = [
            "User_ID",
            "Completion_Status",
            "Engagement_Score",
            "Satisfaction_Score",
            "Completion_Date",
            "Dropout_Reason"
        ]

        new_data.drop(
            columns=drop_columns,
            errors="ignore",
            inplace=True
        )

        # ----------------------------
        # Encode categorical columns
        # ----------------------------

        for col in label_encoders:

            if (
                col != "Completion_Status"
                and col in new_data.columns
            ):

                new_data[col] = label_encoders[col].transform(
                    new_data[col]
                )

        # ----------------------------
        # Prediction
        # ----------------------------

        prediction = model.predict(new_data)

        prediction = label_encoders[
            "Completion_Status"
        ].inverse_transform(prediction)

        # ----------------------------
        # Show Results
        # ----------------------------

        result["Predicted_Status"] = prediction

        st.subheader("Prediction Result")

        st.dataframe(result)

        # ----------------------------
        # Prediction Summary
        # ----------------------------

        st.subheader("Prediction Summary")

        st.write(
            result["Predicted_Status"].value_counts()
        )

        # ----------------------------
        # Download CSV
        # ----------------------------

        csv = result.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "📥 Download Prediction Result",
            data=csv,
            file_name="Prediction_Result.csv",
            mime="text/csv"
        )
