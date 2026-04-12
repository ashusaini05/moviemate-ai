from flask import Flask, request, render_template
from markupsafe import escape
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)

# Hardcoded curated movie dataset as requested
MOVIES = [
    {"id": 1, "title": "The Matrix", "description": "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.", "genre": "Action Sci-Fi"},
    {"id": 2, "title": "Inception", "description": "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.", "genre": "Action Sci-Fi Thriller"},
    {"id": 3, "title": "Interstellar", "description": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.", "genre": "Adventure Drama Sci-Fi"},
    {"id": 4, "title": "Dumb and Dumber", "description": "After a woman leaves a briefcase at the airport terminal, a dumb limo driver and his dumber friend set out on a hilarious cross-country road trip.", "genre": "Comedy"},
    {"id": 5, "title": "The Hangover", "description": "Three buddies wake up from a bachelor party in Las Vegas, with no memory of the previous night.", "genre": "Comedy"},
    {"id": 6, "title": "The Dark Knight", "description": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept physical tests of his ability to fight injustice.", "genre": "Action Crime Drama"},
    {"id": 7, "title": "Avengers: Endgame", "description": "After the devastating events of Infinity War, the Avengers assemble once more in order to reverse Thanos' actions.", "genre": "Action Adventure Sci-Fi"},
    {"id": 8, "title": "Titanic", "description": "A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic.", "genre": "Drama Romance"},
    {"id": 9, "title": "The Notebook", "description": "A poor yet passionate young man falls in love with a rich young woman, giving her a sense of freedom.", "genre": "Drama Romance"},
    {"id": 10, "title": "Gladiator", "description": "A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family.", "genre": "Action Adventure Drama"},
    {"id": 11, "title": "The Shawshank Redemption", "description": "Two imprisoned men bond over a number of years, finding solace and eventual redemption.", "genre": "Drama"},
    {"id": 12, "title": "Superbad", "description": "Two co-dependent high school seniors are forced to deal with separation anxiety after their plan goes awry.", "genre": "Comedy"},
    {"id": 13, "title": "Iron Man", "description": "After being held captive in an Afghan cave, billionaire engineer Tony Stark creates a unique weaponized suit of armor.", "genre": "Action Adventure Sci-Fi"},
    {"id": 14, "title": "Joker", "description": "In Gotham City, mentally troubled comedian Arthur Fleck is disregarded by society. He embarks on a downward spiral of revolution.", "genre": "Crime Drama Thriller"},
    {"id": 15, "title": "Spider-Man: No Way Home", "description": "With Spider-Man's identity now revealed, Peter asks Doctor Strange for help. When a spell goes wrong, foes appear.", "genre": "Action Adventure Sci-Fi"}
]

# Get a list of all titles to show as chips on UI
all_movie_titles = [m['title'] for m in MOVIES]

def clean_text(text):
    """
    Cleans text by removing digits and converting to lowercase.
    Using raw string r'\d' to fix the regex warning.
    """
    cleaned = re.sub(r'\d+', '', text)
    return cleaned.lower()

# Combine 'genre' and 'description' to create feature vectors
combined_features = [clean_text(m['genre'] + " " + m['description']) for m in MOVIES]

# Initialize TF-IDF Vectorizer and calculate the similarity matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(combined_features)
cosine_sim_matrix = cosine_similarity(tfidf_matrix)

def get_recommendations(movie_title, max_results=5):
    """
    Given a movie title, returns the actual matched title and 
    a list of top 'max_results' recommended movies.
    """
    movie_title_lower = movie_title.strip().lower()
    
    # 1. First, try an exact case-insensitive match
    movie_idx = -1
    actual_title = ""
    for i, m in enumerate(MOVIES):
        if m['title'].lower() == movie_title_lower:
            movie_idx = i
            actual_title = m['title']
            break
            
    # 2. If no exact match, try a partial match
    if movie_idx == -1:
        for i, m in enumerate(MOVIES):
            if movie_title_lower in m['title'].lower():
                movie_idx = i
                actual_title = m['title']
                break

    # If still not found, return an error message
    if movie_idx == -1:
        return None, "Movie not found in our database. Please try another one."

    # 3. Get similarity scores for this movie based on its index
    sim_scores = list(enumerate(cosine_sim_matrix[movie_idx]))
    
    # 4. Sort the movies based on similarity scores (highest first)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # 5. Extract the best matches (skip index 0 as it is the movie itself)
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
        # Retrieve and escape input to prevent XSS (using markupsafe fixed import)
        raw_movie_name = request.form.get('movie_name', '')
        searched_movie = escape(raw_movie_name.strip())
        
        if searched_movie:
            result_title, result = get_recommendations(str(searched_movie))
            
            if isinstance(result, str):
                # Result is an error string
                error = result
            else:
                # Success
                actual_title = result_title
                recommendations = result

    return render_template('index.html', 
                           recommendations=recommendations, 
                           error=error, 
                           searched_movie=searched_movie,
                           actual_title=actual_title,
                           all_movies=all_movie_titles)

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
