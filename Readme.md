*Content-Based Recommendation System*

Overview:
This is a simple content-based recommendation system that suggests movies based on a user's text description.

How It Works:
1. Converts movie descriptions and user input into TF-IDF vectors.
2. Computes cosine similarity between the user input and all movie descriptions.
3. Returns the top 5 most relevant movie recommendations.

Dataset:
- The dataset (`movie.csv`) contains a small list of movies and their descriptions.
- It is stored locally and automatically loaded by the script.

Setup:
Ensure you have Python 3.x installed. 

Install Dependencies:
pip install pandas scikit-learn

Run the system:
python content.py "Some user description"

Results:

Input:
python content.py "I love horror and thriller movies"

Output:
🎬 Recommended Movies:


        Title       Similarity Score
    A Quiet Place             0.314493

     Hereditary             0.153930
   
     The Exorcist             0.148988
 
      Get Out             0.147663
      
    Insidious             0.144188
