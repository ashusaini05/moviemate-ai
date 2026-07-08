from flask import Flask, request, render_template
from markupsafe import escape
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import csv
import os
import random

app = Flask(__name__)

MOVIES = []
csv_path = 'imdb_movies.csv'

# Load the dataset dynamically from imdb_movies.csv
if os.path.exists(csv_path):
    try:
        with open(csv_path, mode='r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                # Extract year from date_x (format: MM/DD/YYYY)
                date_str = row.get("date_x", "").strip()
                year_match = re.search(r'\d{4}', date_str)
                year = year_match.group(0) if year_match else "N/A"
                
                # Convert 100-point score to 10-point rating
                score_str = row.get("score", "").strip()
                try:
                    rating = f"{float(score_str) / 10:.1f}" if score_str else "N/A"
                except ValueError:
                    rating = "N/A"

                # Truncate the cast list for clean UI presentation
                crew_raw = row.get("crew", "").strip()
                crew_list = [c.strip() for c in crew_raw.split(",") if c.strip()]
                actors_clean = ", ".join(crew_list[:6])

                MOVIES.append({
                    "id": idx + 1,
                    "title": row.get("names", "").strip(),
                    "genre": row.get("genre", "").strip(),
                    "description": row.get("overview", "").strip(),
                    "rating": rating,
                    "year": year,
                    "actors": actors_clean
                })
        print(f"Successfully loaded {len(MOVIES)} movies from '{csv_path}'.")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
else:
    print(f"Critical Error: '{csv_path}' file not found.")

# Get list of all titles for autocomplete dropdown
all_movie_titles = [m['title'] for m in MOVIES]

def clean_text(text):
    r"""
    Cleans text by removing digits and converting to lowercase.
    Using raw string r'\d' to fix the regex warning.
    """
    cleaned = re.sub(r'\d+', '', text)
    return cleaned.lower()

# Combine features (genre + description + actors) for content-based vectorization
combined_features = []
for m in MOVIES:
    feature_text = f"{m.get('genre', '')} {m.get('description', '')} {m.get('actors', '')}"
    combined_features.append(clean_text(feature_text))

# Initialize TF-IDF Vectorizer and compute sparse matrix (computed once on startup)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(combined_features)

def get_recommendations(movie_title, max_results=5):
    """
    Given a movie title, returns the actual matched title and 
    a list of top 'max_results' recommended movies.
    Optimized: Computes similarity dynamically on-the-fly to save ~800MB of RAM.
    """
    movie_title_lower = movie_title.strip().lower()
    
    # 1. Search for movie (exact match first)
    movie_idx = -1
    actual_title = ""
    for i, m in enumerate(MOVIES):
        if m['title'].lower() == movie_title_lower:
            movie_idx = i
            actual_title = m['title']
            break
            
    # 2. Search for movie (partial match fallback)
    if movie_idx == -1:
        for i, m in enumerate(MOVIES):
            if movie_title_lower in m['title'].lower():
                movie_idx = i
                actual_title = m['title']
                break

    if movie_idx == -1:
        return None, "Movie not found in our database. Please try another one."

    # 3. Optimization: Compute cosine similarity on-the-fly ONLY for the selected movie vector.
    # This prevents storing a massive N x N dense similarity matrix in memory.
    query_vector = tfidf_matrix[movie_idx]
    sim_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # 4. Pair index with score and sort by similarity score (highest first)
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # 5. Get top matches (ignoring index 0 since it is the movie itself)
    best_matches = sim_scores[1:max_results+1]
    
    # 6. Retrieve movie objects
    recommended_movies = [MOVIES[idx] for idx, score in best_matches]
        
    return actual_title, recommended_movies

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    error = None
    searched_movie = ""
    actual_title = ""
    
    if request.method == 'POST':
        # Retrieve and escape input to prevent XSS
        raw_movie_name = request.form.get('movie_name', '')
        searched_movie = escape(raw_movie_name.strip())
        
        if searched_movie:
            result_title, result = get_recommendations(str(searched_movie))
            
            if isinstance(result, str):
                error = result
            else:
                actual_title = result_title
                recommendations = result

    # Display 7 random movie suggestion chips to make the UI dynamic
    suggestion_chips = random.sample(all_movie_titles, min(len(all_movie_titles), 7)) if all_movie_titles else []

    # Sample 8 full movie dictionaries for the home page gallery grid
    popular_movies = random.sample(MOVIES, min(len(MOVIES), 8)) if MOVIES else []

    return render_template('index.html', 
                           recommendations=recommendations, 
                           error=error, 
                           searched_movie=searched_movie,
                           actual_title=actual_title,
                           all_movies=all_movie_titles,
                           suggestion_chips=suggestion_chips,
                           popular_movies=popular_movies)

if __name__ == '__main__':
    # Run the Flask app on port 5001 to avoid conflicts
    app.run(debug=True, port=5001)
