import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns


file_path = "student_feedback.csv"   # use the provided CSV in the workspace
df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)
print(df.head())


# Drop fully-empty rows
df.dropna(how="all", inplace=True)

# If there's an unnamed index-like first column (from CSV leading comma), drop it
first_col = df.columns[0]
if first_col.startswith("Unnamed") or first_col == "":
    df.drop(columns=[first_col], inplace=True)


df.rename(columns=lambda x: x.strip(), inplace=True)

def get_sentiment(text):
    analysis = TextBlob(str(text))
    polarity = analysis.sentiment.polarity
    
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

if "Feedback" in df.columns:
    # Perform sentiment analysis on textual feedback
    df["Sentiment"] = df["Feedback"].apply(get_sentiment)
else:
    print("No textual 'Feedback' column found — skipping sentiment analysis.")
    print("Generating descriptive statistics for numeric response columns instead.")
    print(df.describe())


if "Sentiment" in df.columns:
    sentiment_counts = df["Sentiment"].value_counts()
    print("\nSentiment Counts:\n", sentiment_counts)

    sns.set(style="whitegrid")
    plt.figure(figsize=(6,4))
    sns.countplot(x="Sentiment", data=df)
    plt.title("Student Feedback Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Responses")
    plt.tight_layout()
    plt.savefig("sentiment_counts.png")
    plt.close()
else:
  
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        sns.set(style="whitegrid")
        df[numeric_cols].hist(bins=8, figsize=(10, 8))
        plt.suptitle("Distributions of Numeric Response Columns")
        plt.tight_layout()
        plt.savefig("distributions.png")
        plt.close()

    
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        print("\nCorrelation matrix:\n", corr)

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Correlation Heatmap of Numeric Survey Responses")
        plt.tight_layout()
        plt.savefig("correlation_heatmap.png")
        plt.close()

        # Barplot of mean scores per numeric column (excluding Student ID)
        cols_for_mean = [c for c in numeric_cols if c.lower() not in ("student id", "student_id", "id")]
        if len(cols_for_mean) > 0:
            means = df[cols_for_mean].mean().sort_values(ascending=False)
            plt.figure(figsize=(10,4))
            sns.barplot(x=means.index, y=means.values)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Average Score')
            plt.title('Average Scores per Question')
            plt.tight_layout()
            plt.savefig("mean_scores_bar.png")
            plt.close()

            # Pie chart for a representative question (default to course recommendation)
            default_col = None
            if "Course recommendation based on relevance" in df.columns:
                default_col = "Course recommendation based on relevance"
            elif len(cols_for_mean) > 0:
                default_col = cols_for_mean[0]

            if default_col is not None:
                counts = df[default_col].value_counts().sort_index()
                plt.figure(figsize=(6,6))
                counts.plot.pie(autopct='%1.1f%%', startangle=90)
                plt.ylabel('')
                plt.title(f'Distribution of {default_col}')
                plt.tight_layout()
                plt.savefig("pie_chart.png")
                plt.close()


df.to_csv("Processed_Event_Feedback.csv", index=False)
print("\nProcessed file saved as student_feedback.csv")