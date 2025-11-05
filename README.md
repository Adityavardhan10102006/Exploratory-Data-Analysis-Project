Exploratory Data Analysis (EDA) of Movie Data

Project Overview

This project performs an Exploratory Data Analysis (EDA) on a sample movie dataset (simulated here with core features like Budget, Revenue, and Runtime). The goal is to understand the structure of the data, identify key statistical relationships, and create derived features to assess movie success.

The final outcome is a comprehensive report and a reusable Python script that can be applied to real-world movie datasets (like TMDB).

Key Components

project_da.py: The primary Python script containing the full EDA workflow, from data inspection to advanced visualization.

movie_data_eda_report.md: The formal report summarizing all findings.

Key Findings (from EDA)

Financial Correlation: A very strong positive correlation exists between Budget and Revenue, confirmed visually via a scatter plot.

Feature Engineering (Profit): The derived Profit feature (Revenue - Budget) provides a clearer metric for movie success and highlights high-return investments.

Distribution: All financial metrics (Budget, Revenue, Profit) are heavily right-skewed, indicating the influence of a small number of high-performing blockbuster films (outliers).

Genre Insight: Action and Drama are the most frequently occurring genres in the dataset.

Setup and Execution

Prerequisites

You must have Python and the following libraries installed:

pip install pandas numpy matplotlib seaborn


How to Run the Script

Clone this repository to your local machine:

git clone [https://github.com/Adityavardhan10102006/Exploratory-Data-Analysis-Project.git](https://github.com/Adityavardhan10102006/Exploratory-Data-Analysis-Project.git)
cd Exploratory-Data-Analysis-Project


Execute the Python script:

python project_da.py


The script will print summaries to the console and display all generated charts (histograms, box plots, and heatmaps).
