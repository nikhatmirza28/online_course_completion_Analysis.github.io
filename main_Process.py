#################################################################################################################
## Project Title:-Online Course Completion Rate Analysis
## Project Description:-This project aims to analyze the completion rates of online courses offered by various platforms.
## The analysis will help identify factors that influence course completion and provide insights for improving course design and delivery.  
## The project will involve data collection, cleaning, and analysis using statistical and machine learning techniques.
###################################################################################################################
#1.Loading Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly
import seaborn as sns
import plotly.express as px
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,confusion_matrix,classification_report)
import joblib
"""
##############################################
#2. Loading the Dataset and Basic Information
##############################################
df=pd.read_csv('online_learning_course_completion_final.csv')
print("Dataset Loaded Successfully")
print(df.info())
print(df.shape)
print(df.columns)
print(df.describe())
##############################################
#3. Data Cleaning and Preprocessing
##############################################
# Checking for missing values
print(df.isnull().sum())
#Completion_date column has 344 missing values out of 500 records that is 68.8% missing values.Now we have to find out the reason for this missing values and how to handle them.

df["Completion_Date"] = pd.to_datetime(df["Completion_Date"])
print(df["Completion_Date"].isnull().sum())

#duplicate values
print(df.duplicated().sum())

# Convert dates
df["Enrollment_Date"] = pd.to_datetime(df["Enrollment_Date"])
df["Completion_Date"] = pd.to_datetime(df["Completion_Date"])

# Calculate completion days
#create a new column "Completion_Days" to calculate the number of days taken to complete the course by subtracting the "Enrollment_Date" from the "Completion_Date". The result is converted to days using the dt.days attribute.   

df["Completion_Days"] = ( df["Completion_Date"] - df["Enrollment_Date"]).dt.days
print(df["Completion_Days"].head())

###########################################
#4. Exploratory Data Analysis (EDA)
###########################################

print(df.describe())

###############Univariate Analysis###############

#4.1.Experience Level Distribution
print(df["Experience_Level"].value_counts())
#insights from the Experience_Level distribution:
#• The dataset contains a mix of working professionals, freshers, and students.
#Experience_Level
#Working Professional    173
#Fresher                 171
#Student                 156
#•The number of working professionals is slightly higher than freshers and students, indicating that the courses are attracting a diverse audience.

#4.2 category column:
print(df["Category"].value_counts())

#vizualization:

plt.figure(figsize=(10,5))
sns.countplot(data=df, x="Category", palette="viridis")
plt.title("Course Category Distribution")
plt.xlabel("Course Category")
plt.ylabel("Number of Students")
plt.xticks(rotation=45)
plt.show()
#insights from the Category distribution:
#The dataset contains courses from different categories such as Data Science,sql,power bi,Statistics,AI,Tableau,Python and Machine Learning.
#Data Science has the highest number of enrolled students, indicating strong learner interest.
#Machine Learning and Python also attract a significant number of learners.
#Categories with fewer enrollments may require better promotion or updated course content.

#4.3 Platform Distribution
print(df["Platform"].value_counts())
#visualization:
plt.figure(figsize=(10,5))
sns.countplot(data=df,x="Platform",palette="magma")
plt.title("Platform Distribution")
plt.xlabel("Platform")
plt.ylabel("Number of Students")
plt.xticks(rotation=45)
plt.show()

#insights from the Platform distribution:
#The dataset includes courses from various online learning platforms such as Coursera, Udemy, edX, Skillshare and Youtube.
#Coursera has the highest number of enrolled students, indicating its popularity and wide reach.
#Udemy and edX also attract a significant number of learners, while Skillshare and Youtube have fewer enrollments.



#4.4 Extract Enrollment Month and Year

df["Enrollment_Month"] = df["Enrollment_Date"].dt.month_name()

df["Enrollment_Year"] = df["Enrollment_Date"].dt.year


print(df['Enrollment_Month'].value_counts())

month_order = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"]

#visualization:
plt.figure(figsize=(10,5))

sns.countplot(
    data=df,
    x="Enrollment_Month",
    order=month_order,
    palette="crest"
)

plt.title("Enrollment Month Distribution")
plt.xlabel("Enrollment Month")
plt.ylabel("Number of Students")
plt.xticks(rotation=45)
plt.show()

#insights from the Enrollment Month distribution:
#• Student enrollments are distributed throughout the year, indicating continuous interest in online learning.
#• March and April recorded the highest number of enrollments, suggesting higher learner participation during these months.
#• Enrollment gradually decreases after April, with comparatively fewer enrollments during July to December.
#• The observed monthly variation can help online learning platforms plan course launches and promotional campaigns during high-demand periods.

#4.5 Completion Status Distribution:
print(df["Completion_Status"].value_counts())
plt.figure(figsize=(8,5))
sns.countplot(data=df,x="Completion_Status", order=["Completed", "In Progress", "Dropped"],
    palette="Set2"
)

plt.title("Completion Status Distribution")
plt.xlabel("Completion Status")
plt.ylabel("Number of Students")

plt.show()
#insights from the Completion Status distribution:
#• The dataset shows a balanced distribution of completed, in-progress, and dropped students.
#• The number of dropped students is slightly higher than the number of completed students, indicating opportunities to improve learner retention.
#• A significant number of students are still in progress, suggesting active participation and the potential for future course completions.
#• Improving learner engagement and support may help convert in-progress learners into successful course completions.


#4.6 Completion Percentage Distribution:
print(df["Completion_Percentage"].value_counts())
plt.figure(figsize=(8,5))
sns.histplot( data=df,x="Completion_Percentage",bins=10,kde=True,color="steelblue")
plt.title("Completion Percentage Distribution")
plt.xlabel("Completion Percentage (%)")
plt.ylabel("Number of Students")
plt.show()

#insights from the Completion Percentage distribution:
#Students have different completion percentages, showing different levels of learning progress.
#Most students have completed around 25% to 45% of the course.
#Some students have completed more than 80% of the course, showing good commitment and interest.
#The platform should encourage students who are in the early stages to help them complete the course.
###################################################
#5. Bivariate Analysis:
###################################################
#5.1 Completion Status vs Experience Level
plt.figure(figsize=(8,5))

sns.countplot(data=df,x="Experience_Level",hue="Completion_Status",
    palette="Set2"
)

plt.title("Completion Status by Experience Level")
plt.xlabel("Experience Level")
plt.ylabel("Number of Students")
plt.show()
#print(df["Experience_Level"].value_counts(dropna=False))
#print(df["Experience_Level"].unique())

#Insights from the Completion Status by Experience Level:
# • Students show the highest number of completed courses compared to freshers and working professionals.
# • Freshers have the highest number of dropped and in-progress courses, indicating that they may need additional learning support.
# • Working professionals have a balanced distribution across completed, dropped, and in-progress courses.
# • Experience level appears to influence course completion, suggesting that learner background plays an important role in online learning success.

#5.2 Completion Status vs Category

plt.figure(figsize=(12,6))

sns.countplot(data=df, x="Category",hue="Completion_Status", palette="Set2")
plt.title("Completion Status by Course Category")
plt.xlabel("Course Category")
plt.ylabel("Number of Students")
plt.xticks(rotation=45)
plt.show()

category_completion = pd.crosstab( df["Category"], df["Completion_Status"])
print(category_completion)

#This graph compares completion status across different course categories. It helps identify which courses have better completion rates and which courses require improvement.
#Such analysis can help online learning platforms improve course quality and learner support.

#5.3 Engagement Score vs Completion Status
df.groupby("Completion_Status")["Engagement_Score"].describe()

plt.figure(figsize=(8,5))

sns.boxplot(
    data=df,
    x="Completion_Status",
    y="Engagement_Score",
    palette="Set2"
)

plt.title("Engagement Score by Completion Status")
plt.xlabel("Completion Status")
plt.ylabel("Engagement Score")

plt.show()

#Insights from the Engagement Score by Completion Status::
#"This graph shows that engagement plays an important role in course completion."
#"Students with higher engagement scores tend to complete courses more successfully,"
#"while students with lower engagement are more likely to drop out."

#5.4 Satisfaction Score vs Completion Status
#Statistics:
df.groupby("Completion_Status")["Satisfaction_Score"].describe()
#visualization:

plt.figure(figsize=(8,5))

sns.boxplot(
    data=df,
    x="Completion_Status",
    y="Satisfaction_Score",
    palette="Set2"
)

plt.title("Satisfaction Score by Completion Status")
plt.xlabel("Completion Status")
plt.ylabel("Satisfaction Score")

plt.show()
#insights:
#"The graph shows that students who completed the course generally have higher satisfaction scores, "
#while dropped students have lower satisfaction scores. "
#This indicates that learner satisfaction plays an important role in course completion."


############################################
# 6. Multivariate Analysis
# Correlation Heatmap
#############################################

plt.figure(figsize=(10,8))

# Selected only numerical columns
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Correlation matrix
corr_matrix = numeric_df.corr()

# Heatmap
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap")
plt.xticks(rotation=45)

plt.show()

#Insights:
#Videos Watched and Assignments Submitted have a strong positive correlation (0.79), indicating that students who watch more videos are also more likely to submit assignments.
#Engagement Score and Satisfaction Score show a strong positive correlation (0.78), suggesting that highly engaged students are generally more satisfied with the course.
#Videos Watched and Quiz Score have a strong positive correlation (0.77), indicating that active learners tend to achieve better quiz performance.
#Completion Percentage has a moderate positive correlation with Engagement Score (0.59) and Satisfaction Score (0.55), showing that engaged and satisfied students are more likely to complete a larger portion of the course.
#Hours Spent Per Week, Course Duration, and Completion Days have very weak correlations with most variables, indicating that learner engagement and participation have a greater influence on course completion than time-related factors.



"""
###########################################
# 7. Machine Learning
###########################################

df=pd.read_csv('online_learning_course_completion_final.csv')
df1 = df.copy()



# Convert Enrollment_Date to datetime

df1["Enrollment_Date"] = pd.to_datetime(df1["Enrollment_Date"])

# Create new features

df1["Enrollment_Month"] = df1["Enrollment_Date"].dt.month

df1["Enrollment_Day"] = df1["Enrollment_Date"].dt.day

# Drop original date column

df1.drop(columns=["Enrollment_Date"], inplace=True)

#from sklearn.preprocessing import LabelEncoder

label_encoders = {}

for col in df1.select_dtypes(include="object").columns:

    if col != "User_ID":

        le = LabelEncoder()

        df1[col] = le.fit_transform(df1[col])

        label_encoders[col] = le

        X = df1.drop(
    [
         "User_ID",
         "Completion_Status",
         "Engagement_Score",
         "Satisfaction_Score",
         "Completion_Date",
         "Dropout_Reason"
    ],
    axis=1,
    errors="ignore"
)

y = df1["Completion_Status"]

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20,random_state=42, stratify=y)

print("Training Data :", X_train.shape)

print("Testing Data :", X_test.shape)



# ==========================================
# Train Random Forest Model
# ==========================================

model = RandomForestClassifier(

    n_estimators=100,

    random_state=42

)

model.fit(X_train, y_train)

print("Model Trained Successfully!")



# ==========================================
# Save Model
# ==========================================

import joblib

joblib.dump(model, "course_completion_model.pkl")

print("✅ Model saved successfully!")

joblib.dump(label_encoders, "label_encoders.pkl")

print("✅ Encoders saved successfully!")


# ==========================================
# Prediction
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# Accuracy
# ==========================================


accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy : {accuracy*100:.2f}%")

print(X.columns.tolist())
