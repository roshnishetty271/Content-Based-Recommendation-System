import pandas as pd
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset (Ensure 'movies.csv' exists in the same directory)
def load_data(file_path="movie.csv"):
    df = pd.read_csv(file_path, encoding="ISO-8859-1")
    return df

# Preprocess and compute TF-IDF similarity
def recommend_movies(user_input, df, top_n=5):
    # Combine user input with dataset descriptions
    descriptions = df['Description'].tolist()
    descriptions.append(user_input)

    # Convert text into TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(descriptions)

    # Compute cosine similarity (last row is user input)
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    # Get top N similar items
    top_indices = cosine_sim.argsort()[-top_n:][::-1]  # Sort in descending order
    recommendations = df.iloc[top_indices].copy()
    recommendations["Similarity Score"] = cosine_sim[top_indices]  # Add similarity score

    return recommendations[['Title', 'Similarity Score']]

# Main execution
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python content.py 'Your movie preferences here'")
        sys.exit(1)

    user_input = sys.argv[1]
    df = load_data()
    results = recommend_movies(user_input, df)

    print("\n🎬 Recommended Movies:\n")
    print(results.to_string(index=False))  # Display movies in table format
