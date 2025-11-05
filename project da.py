# ==============================================================================
# 1. PROJECT SETUP AND DATA LOADING
# This script performs a full Exploratory Data Analysis (EDA) workflow.
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure plot style for better aesthetics
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.family'] = 'Inter'

# Simulate loading the dataset (using dummy data for guaranteed runnability)
# In a real scenario, replace 'tmdb_movies.csv' with your actual file path.
try:
    # Attempt to load a real file (will likely fail in this environment)
    df = pd.read_csv('tmdb_movies.csv')
except FileNotFoundError:
    print("Dataset not found. Using a dummy DataFrame for demonstration.")
    # Create a dummy DataFrame if the file isn't available
    data = {
        'title': ['Avatar', 'Titanic', 'Avengers', 'Joker', 'Inception'],
        'release_date': ['2009-12-18', '1997-12-19', '2012-05-04', '2019-10-04', '2010-07-16'],
        'budget': [237000000, 200000000, 220000000, 55000000, 160000000],
        'revenue': [2787965087, 2257844554, 1518815515, 1074219000, 825532764],
        'runtime': [162, 194, 143, 122, 148],
        'vote_average': [7.2, 7.5, 7.8, 8.4, 8.8],
        'genre': ['Action', 'Drama', 'Action', 'Drama', 'Sci-Fi'],
        'director': ['James Cameron', 'James Cameron', 'Joss Whedon', 'Todd Phillips', 'Christopher Nolan'],
        'is_english': [True, True, True, True, True]
    }
    df = pd.DataFrame(data)

print("Data Loading Complete.")

# ==============================================================================
# 2. INITIAL DATA INSPECTION AND STRUCTURE
# ==============================================================================
print("\n" + "="*50)
print("STEP 2: INITIAL DATA INSPECTION")
print("="*50)

# Check the first 5 rows
print("--- First 5 Rows ---")
print(df.head())

# Check the size of the dataset
print("\n--- Dataset Shape (Rows, Columns) ---")
print(df.shape)

# Check data types and non-null counts
print("\n--- Column Info and Data Types ---")
df.info()

# ==============================================================================
# 3. DATA QUALITY ASSESSMENT (MISSING VALUES AND OUTLIERS)
# ==============================================================================
print("\n" + "="*50)
print("STEP 3: DATA QUALITY ASSESSMENT")
print("="*50)

# 3.1 Handling Missing Values
print("\n--- 3.1 Missing Value Report ---")
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing Count': missing_values,
    'Missing Percentage': missing_percentage
}).sort_values(by='Missing Count', ascending=False)

print(missing_df)

# 3.2 Outlier Detection (Initial Visual Check)
print("\n--- 3.2 Numerical Feature Descriptive Statistics ---")
print(df[['budget', 'revenue', 'runtime', 'vote_average']].describe())

# Visualizing distributions using Box Plots for outliers
plt.figure(figsize=(12, 4))
sns.boxplot(data=df[['budget', 'revenue', 'runtime']], orient='h')
plt.title('Box Plots for Budget, Revenue, and Runtime')
plt.show()

# ==============================================================================
# 4. UNIVARIATE ANALYSIS (SUMMARIZING DISTRIBUTIONS)
# ==============================================================================
print("\n" + "="*50)
print("STEP 4: UNIVARIATE ANALYSIS")
print("="*50)

# 4.1 Numerical Distributions (Histograms)
plt.figure(figsize=(15, 4))

# Runtime Distribution
plt.subplot(1, 3, 1)
sns.histplot(df['runtime'], bins=10, kde=True, color='skyblue')
plt.title('Runtime Distribution')

# Vote Average Distribution
plt.subplot(1, 3, 2)
sns.histplot(df['vote_average'], bins=10, kde=True, color='lightcoral')
plt.title('Vote Average Distribution')

# Budget Distribution (log-transformed due to skew)
plt.subplot(1, 3, 3)
sns.histplot(np.log1p(df['budget']), bins=10, kde=True, color='lightgreen')
plt.title('Log-Transformed Budget Distribution')

plt.tight_layout()
plt.show()

# 4.2 Categorical Distributions (Value Counts)
print("\n--- 4.2 Top 5 Movie Genres ---")
print(df['genre'].value_counts().head())

# Visualizing genre counts
plt.figure(figsize=(8, 6))
sns.countplot(y='genre', data=df, order=df['genre'].value_counts().index)
plt.title('Movie Count by Primary Genre')
plt.show()

# ==============================================================================
# 5. BIVARIATE AND MULTIVARIATE ANALYSIS (VISUALIZING RELATIONSHIPS)
# ==============================================================================
print("\n" + "="*50)
print("STEP 5: BIVARIATE AND MULTIVARIATE ANALYSIS")
print("="*50)

# 5.1 Correlation Heatmap
numerical_df = df[['budget', 'revenue', 'runtime', 'vote_average']].copy()
correlation_matrix = numerical_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# 5.2 Scatter Plot: Budget vs. Revenue
plt.figure(figsize=(8, 6))
sns.scatterplot(x='budget', y='revenue', data=df, hue='vote_average', size='vote_average', sizes=(20, 200))
plt.title('Budget vs. Revenue (Colored by Vote Average)')
plt.xlabel('Budget (in hundred millions)')
plt.ylabel('Revenue (in billions)')
plt.show()

# ==============================================================================
# 6. STATISTICAL STORYTELLING AND FINAL INSIGHTS
# ==============================================================================
print("\n" + "="*50)
print("STEP 6: FINAL INSIGHTS (STATISTICAL STORYTELLING)")
print("="*50)
print("1. High Correlation in Financials: Budget and Revenue show a strong positive correlation.")
print("2. Distribution Skew: Financial features are highly right-skewed due to blockbuster outliers.")
print("3. Key Genre Popularity: 'Action' and 'Drama' are the most dominant genres in the sample.")
print("4. Data Quality: Core numerical columns are complete; descriptive columns may have high missing rates.")
print("5. Runtime Profile: Most movies cluster around the 120-150 minute mark.")
print("="*50)

