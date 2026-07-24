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

    import streamlit as st

    st.title("📚Online Course Completion Prediction System")
    st.markdown("This project uses Machine Learning to predict whether a student will Complete, Drop or remain In Progress based on learning behaviour.")

    st.image( "assets/stremlit_homepage.png", use_container_width=True )

    st.markdown("---")

    st.subheader("Project Overview")

    st.write("""This project predicts whether a learner will Complete, Drop, or remain In Progress using Machine Learning. It also provides interactive Power BI dashboards to analyze learner behavior and performance.""")

    st.markdown("## ⚠️ Problem Statement")

    st.write("""
        Many students enroll in online courses but do not complete them. Online learning platforms
        find it difficult to identify learners who are likely to complete, continue, or drop out.
        This project addresses this challenge by predicting learner completion status and providing
        analytical insights to improve student engagement and course completion.
        """)

    st.subheader("Project Objective")
    st.write("""
        The main objective of this project is to predict the completion status of learners using
        Machine Learning. The system classifies learners as Completed, In Progress, or Dropped,
        helping online learning platforms analyze student performance and make data-driven decisions
        to improve learner retention and success.
        """)
    st.subheader("Project Feature")
    st.write ("""Key Features: 
    
        ✅ Course Completion Prediction

        ✅ Interactive Power BI Dashboard

        ✅ Student Performance Analytics

        ✅ Platform-wise Analysis

        ✅ Course Category Analysis

        ✅ Batch Prediction""")

    st.markdown("## 📂 Dataset")

    st.info("""
        **Dataset Source:** Kaggle

        **Dataset Name:** Online Learning & Course Consumption Dataset

        The original dataset was collected from Kaggle and later cleaned,
        preprocessed, and enhanced with additional features for Machine Learning
        model training and Power BI dashboard development.

        Dataset Link:
        https://www.kaggle.com/datasets/prince7489/online-learning-and-course-consumption-dataset
        """)
    st.subheader("Dataset Preview")

    st.dataframe(df.head())


    
# ==========================================
# ANALYSIS PAGE
# ==========================================

elif page == "📊 Analysis":

    st.title("📊 Exploratory Data Analysis")

    st.write("The following visualizations provide insights into learner behaviour and course completion patterns.")

    col1, col2 = st.columns(2)


        
    with col1:
                    st.subheader("Course Category Distribution")
                    st.image("screenshots/Course_Catagory.JPG", use_container_width=True)
            
    with col2:
                    st.subheader("Plateform_Distribution")
                    st.image("screenshots/Plateform_Dist.JPG", use_container_width=True)


    
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Completion Percentage")
        st.image("screenshots/Completion_percentage.JPG", use_container_width=True)

    with col4:
        st.subheader("Engagement vs Completion")
        st.image("screenshots/EngatementScoreVSCompleStat.JPG", use_container_width=True)



    col5, col6 = st.columns(2)

    with col5:
            st.subheader("Completion Status")
            st.image("screenshots/Completion_status.JPG", use_container_width=True)
    
    with col6:
            st.subheader("Enrollment Month")
            st.image("screenshots/Enrollment_month.JPG", use_container_width=True)


    
    col7, col8 = st.columns(2)
    
    with col7:
                st.subheader("Satisfaction Score By Completion status")
                st.image("screenshots/SatisfactionScoreVSCompletion.JPG", use_container_width=True)
        
    with col8:
                st.subheader("Completion Status by Experience Level")
                st.image("screenshots/CompletionVSExpLevel.JPG", use_container_width=True)

    st.subheader("Correlation Heatmap")

    st.image("screenshots/HeatMap (2).JPG", use_container_width=True)


    st.title (" Power BI Dashboard ")
    
    st.markdown("---")
    st.write("""
        The following Power BI dashboards provide detailed insights into
        student enrollment, course completion, learner engagement,
        platform performance, certifications, and overall learning trends.
        """)

    st.markdown("### Dashboard - Page 1")

    st.image(
    "dashboards/dashboard_page1_new.JPG",
    use_container_width=True
        )

    st.markdown("### Dashboard - Page 2")

    st.image(
    "dashboards/Dashbord_Page2.JPG",
    use_container_width=True
)

# ==========================================
# BATCH PREDICTION PAGE
# ==========================================

    st.set_page_config(
        page_title="Online Course Completion Prediction",
        page_icon="🎓",
        layout="wide"
      )
    
    st.title("🤖 Batch Prediction")
    st.image( "assets/last page.png",use_container_width=True
                  )

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
