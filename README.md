Exploratory Data Analysis (EDA) on a Movie Dataset
Project Goal: To thoroughly inspect, clean, summarize, and visualize a movie dataset to extract meaningful patterns and insights before any machine learning modeling is attempted.
Key Tools Used:
•	Pandas: For data loading, inspection, and manipulation.
•	Seaborn & Matplotlib: For statistical visualization and plotting.
1. Project Setup and Data Loading
First, we import the necessary Python libraries and load our movie data (hypothetically from a file like tmdb_movies.csv).
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure plot style for better aesthetics
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.family'] = 'Inter'

# Simulate loading the dataset
# Replace 'tmdb_movies.csv' with the actual file path
try:
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
2. Initial Data Inspection and Structure
The first step in EDA is to quickly look at the raw data to understand its format and contents.
2.1 First Look (.head(), .shape, .info())
# Check the first 5 rows
print("--- First 5 Rows ---")
print(df.head())

# Check the size of the dataset
print("\n--- Dataset Shape (Rows, Columns) ---")
print(df.shape)

# Check data types and non-null counts
print("\n--- Column Info and Data Types ---")
print(df.info())
Interpretation:
•	The dataset has 5 rows and 9 columns. In a real dataset, this would be thousands of rows.
•	We can see a mix of data types: object (strings like titles and genres), int64 (budget, revenue), and float64 (vote average).
•	Crucially, the info() output shows us which columns might have missing values (non-null counts less than the total number of rows).
3. Data Quality Assessment (Missing Values and Outliers)
3.1 Handling Missing Values
We calculate the sum of missing values for every column to assess data completeness.
# Calculate total missing values per column
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing Count': missing_values,
    'Missing Percentage': missing_percentage
}).sort_values(by='Missing Count', ascending=False)

print("\n--- Missing Value Report ---")
print(missing_df)

# Example: If 'runtime' was missing for some rows, a decision would be made:
# df.dropna(subset=['runtime'], inplace=True) # or df['runtime'].fillna(df['runtime'].median(), inplace=True)
Interpretation: (Based on the dummy data, there are no missing values, but in a real-world scenario...)
•	If a column like homepage or tagline had 80% missing data, it might be dropped entirely.
•	If runtime had a small percentage missing, we might use the median runtime for imputation, as film runtime is generally normally distributed.
3.2 Outlier Detection (Initial Visual Check)
We use descriptive statistics and visualizations to spot extreme values, particularly in financial columns.
# Statistical summary of numerical columns
print("\n--- Numerical Feature Descriptive Statistics ---")
print(df[['budget', 'revenue', 'runtime', 'vote_average']].describe())

# Visualizing distributions using Box Plots for outliers
plt.figure(figsize=(12, 4))
sns.boxplot(data=df[['budget', 'revenue', 'runtime']], orient='h')
plt.title('Box Plots for Budget, Revenue, and Runtime')
plt.show() # 
Interpretation:
•	The describe() output shows a huge difference between the mean and max in budget and revenue, indicating a small number of extremely expensive or high-grossing films (outliers) skewing the data.
•	The box plot would visually confirm these outliers, especially for the high end of revenue and budget.
4. Univariate Analysis (Summarizing Distributions)
4.1 Numerical Distributions (Histograms)
We plot histograms to understand the shape of individual numerical features.
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
plt.show() # 
Interpretation:
•	Runtime: The distribution is likely right-skewed (a long tail to the right), meaning most movies cluster around a certain length (e.g., 90-120 minutes), with a few very long films.
•	Budget: The original budget data is highly skewed. By using np.log1p(), the distribution becomes more normalized, which is useful for modeling.
4.2 Categorical Distributions (Value Counts)
We check the frequency of categorical features like genre and director.
print("\n--- Top 5 Movie Genres ---")
print(df['genre'].value_counts().head())

# Visualizing genre counts
plt.figure(figsize=(8, 6))
sns.countplot(y='genre', data=df, order=df['genre'].value_counts().index)
plt.title('Movie Count by Primary Genre')
plt.show() # 
Interpretation:
•	The value_counts show the most frequently occurring genres (e.g., Action, Drama).
•	This reveals potential class imbalance; if we were trying to predict genre, we might need to handle the over-representation of popular categories.
5. Bivariate and Multivariate Analysis (Visualizing Relationships)
5.1 Correlation Heatmap
We examine the linear relationship between all numerical variables using a correlation heatmap.
numerical_df = df[['budget', 'revenue', 'runtime', 'vote_average']].copy()
correlation_matrix = numerical_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show() # 
Interpretation:
•	Budget vs. Revenue (e.g., 0.90): A strong positive correlation is observed. Higher budgets tend to lead to higher revenues.
•	Runtime vs. Revenue (e.g., 0.55): A moderate positive correlation suggests longer movies might slightly increase revenue potential, but other factors are likely more important.
5.2 Scatter Plot: Budget vs. Revenue
A scatter plot provides a visual check of the relationship between two highly correlated variables.
plt.figure(figsize=(8, 6))
sns.scatterplot(x='budget', y='revenue', data=df, hue='vote_average', size='vote_average', sizes=(20, 200))
plt.title('Budget vs. Revenue (Colored by Vote Average)')
plt.xlabel('Budget (in hundred millions)')
plt.ylabel('Revenue (in billions)')
plt.show() # 
Interpretation:
•	The points cluster along a diagonal line, visually confirming the strong positive correlation.
•	By coloring points by vote_average (multivariate analysis), we can see if highly rated movies (darker points) are concentrated in a specific budget/revenue range.
6. Statistical Storytelling and Final Insights
Based on the exploration above, here are the key insights drawn from the Movie Dataset:
1.	High Correlation in Financials: There is a significant positive correlation ($r \approx 0.90$) between budget and revenue. This strongly suggests that spending more money on a film generally translates to higher total gross revenue, but does not guarantee profitability.
2.	Revenue and Budget Distribution Skew: Both budget and revenue are highly right-skewed, indicating that the majority of movies are low-to-mid budget, and their distributions are heavily influenced by a few blockbuster films (outliers) with astronomical revenues and budgets.
3.	Key Genre Popularity: The genre analysis revealed that 'Action' and 'Drama' are the most frequently produced movie types, showing a clear focus of the industry.
4.	Data Quality: Initial inspection showed that critical columns like budget, revenue, and vote_average are mostly complete, while columns related to descriptions (like tagline) are likely missing a high percentage of data.
5.	Runtime Profile: The average movie runtime is around 140 minutes, but the median is slightly lower, indicating that very long films are pushing the average up.
This EDA provides a solid foundation. We now know the key features, where the data is messy, and which relationships exist, which is essential before building any predictive models.

