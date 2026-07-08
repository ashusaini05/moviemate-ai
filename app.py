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

def get_recommendations(movie_title, genre_filter="", sort_by="similarity", max_results=5):
    """
    Given a movie title, returns the actual matched title and 
    a list of top 'max_results' recommended movies.
    Supports filtering by genre and sorting by rating/similarity.
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
    
    # Pair indices with similarity score
    indexed_scores = list(enumerate(sim_scores))
    
    # Filter by genre if requested
    if genre_filter:
        genre_filter_lower = genre_filter.strip().lower()
        indexed_scores = [
            (idx, score) for idx, score in indexed_scores 
            if idx == movie_idx or genre_filter_lower in MOVIES[idx]['genre'].lower()
        ]

    # Sort results
    if sort_by == "rating":
        def get_rating_val(idx):
            try:
                return float(MOVIES[idx]['rating'])
            except ValueError:
                return 0.0
        indexed_scores = sorted(indexed_scores, key=lambda x: (get_rating_val(x[0]), x[1]), reverse=True)
    else:
        indexed_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)

    # Get top matches (ignoring the movie itself)
    best_matches = []
    for idx, score in indexed_scores:
        if idx == movie_idx:
            continue
        best_matches.append((idx, score))
        if len(best_matches) >= max_results:
            break
            
    recommended_movies = [MOVIES[idx] for idx, score in best_matches]
    return actual_title, recommended_movies

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    error = None
    searched_movie = ""
    actual_title = ""
    genre_filter = ""
    sort_by = "similarity"
    
    # Extract unique genres from all movies to populate the genre filter dropdown
    genres_set = set()
    for m in MOVIES:
        for g in m['genre'].split(','):
            g_clean = g.strip()
            if g_clean:
                genres_set.add(g_clean)
    all_genres = sorted(list(genres_set))

    if request.method == 'POST':
        # If it is a "Surprise Me" action, pick a random movie title
        if request.form.get('surprise_me') == 'true':
            if all_movie_titles:
                searched_movie = random.choice(all_movie_titles)
        else:
            raw_movie_name = request.form.get('movie_name', '')
            searched_movie = raw_movie_name.strip()
            
        searched_movie = escape(searched_movie)
        genre_filter = request.form.get('genre_filter', '').strip()
        sort_by = request.form.get('sort_by', 'similarity').strip()
        
        if searched_movie:
            result_title, result = get_recommendations(str(searched_movie), genre_filter, sort_by)
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
                           popular_movies=popular_movies,
                           all_genres=all_genres,
                           selected_genre=genre_filter,
                           selected_sort=sort_by)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
