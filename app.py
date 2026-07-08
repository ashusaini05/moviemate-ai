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
            for idx, row in enumerate(csv.DictReader(f)):
                # Extract year (format: MM/DD/YYYY)
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
        print(f"Successfully loaded {len(MOVIES)} movies.")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
else:
    print(f"Critical Error: '{csv_path}' file not found.")

all_movie_titles = [m['title'] for m in MOVIES]

# Combine features (genre + description + actors) for content-based vectorization
combined_features = [f"{m['genre']} {m['description']} {m['actors']}".lower() for m in MOVIES]

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
    
    # Match movie (exact match first, then partial match fallback)
    matches = [i for i, m in enumerate(MOVIES) if m['title'].lower() == movie_title_lower]
    if not matches:
        matches = [i for i, m in enumerate(MOVIES) if movie_title_lower in m['title'].lower()]
        
    if not matches:
        return None, "Movie not found in our database. Please try another one."
        
    movie_idx = matches[0]
    actual_title = MOVIES[movie_idx]['title']

    # Compute cosine similarity on-the-fly ONLY for the selected movie vector.
    query_vector = tfidf_matrix[movie_idx]
    sim_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get top matches (ignoring index 0 since it is the movie itself)
    best_matches = sorted(list(enumerate(sim_scores)), key=lambda x: x[1], reverse=True)[1:max_results+1]
    recommended_movies = [MOVIES[idx] for idx, score in best_matches]
        
    return actual_title, recommended_movies

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    error = None
    searched_movie = ""
    actual_title = ""
    
    if request.method == 'POST':
        raw_movie_name = request.form.get('movie_name', '')
        searched_movie = escape(raw_movie_name.strip())
        
        if searched_movie:
            result_title, result = get_recommendations(str(searched_movie))
            if isinstance(result, str):
                error = result
            else:
                actual_title = result_title
                recommendations = result

    suggestion_chips = random.sample(all_movie_titles, min(len(all_movie_titles), 7)) if all_movie_titles else []
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
    app.run(debug=True, port=5001)
