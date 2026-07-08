# Movie Recommendation System - Project Report

## 1. Project Overview
This project is a **Content-Based Movie Recommendation System** built as a lightweight web application. The core objective of the system is to allow a user to search for a movie and receive 5 highly relevant movie recommendations based on textual similarity (genres, descriptions, directors, and actors) computed dynamically from a dataset of 1,000+ films, without the need for an active database setup.

---

## 2. Technology Stack
The project was designed to be fast, minimal, and beginner-friendly, keeping external dependencies low.
- **Programming Language:** Python 3.11+
- **Backend Web Framework:** Flask
- **Machine Learning / NLP:** Scikit-Learn (`TfidfVectorizer`, `cosine_similarity`)
- **Frontend / UI:** HTML5, Vanilla CSS
- **Data Storage:** Flat CSV file (`movies.csv`) parsed dynamically with Python's standard `csv` module (with an in-memory fallback list).

---

## 3. Original Errors & How They Were Solved

During the initial phase of the project, the legacy codebase crashed due to several critical errors. Here is how they were resolved:

### Error 1: Flask / Werkzeug Import Error
- **The Issue:** `ImportError: cannot import name 'escape' from flask`. In newer versions of Flask and Werkzeug, the `escape` function was deprecated and moved out of the main Flask module.
- **The Solution:** We updated the import statement to specifically use `from markupsafe import escape`, restoring security against XSS (Cross-Site Scripting) attacks without breaking the app.

### Error 2: Missing & Broken Database Requirements
- **The Issue:** The project previously relied on MySQL via `pymysql` and a heavy data access layer (`dbo.py`). The project completely crashed without a properly configured SQL environment on the local machine.
- **The Solution:** We **removed MySQL entirely**. The data access layers (`dbo.py`, `doMovie.py`, `doMovies.py`) were deleted. We migrated to a CSV-based dataset loading system (`movies.csv`) which runs instantly on any machine without SQL installation.

### Error 3: Improper NLP Library Usage
- **The Issue:** The system previously relied on `jieba`, an external library designed for tokenizing Chinese text, which made no sense for an English-based dataset.
- **The Solution:** We scrapped `jieba` completely and replaced it with `scikit-learn`'s robust `TfidfVectorizer` to handle English stop-words and vectorization efficiently.

### Error 4: Python 3.11+ Regex Warnings
- **The Issue:** `SyntaxWarning: invalid escape sequence \d`. Newer versions of Python throw a syntax warning when backslashes are used improperly in normal strings.
- **The Solution:** Converted the regular expression string to a raw string format (`r'\d+'`), strictly adhering to modern Python best practices.

---

## 4. System Approach & Methodology
The project implements a **Content-Based Filtering** recommendation algorithm. 

1. **Feature Engineering:** When the application starts, it creates a "bag of words" for each movie by combining the `genre`, `description`, `director`, and `actors` of the movie into a single continuous string. Including directors and actor cast members significantly improves similarity mapping.
2. **Data Cleaning:** Using Python's `re` (Regex) module, we normalize the text by enforcing lowercase and stripping away numbers and trailing spaces.
3. **User Flow & Autocomplete:** A user enters a movie name into the text field. To handle the scale of 1,000+ movies, we implemented a native HTML `<datalist>` autocomplete dropdown. This lets the user select movies easily without spelling mistakes.
4. **Dynamic chips:** To make the user interface feel alive, we display 7 random suggestion chips from the dataset each time the user loads or refreshes the page.

---

## 5. Which AI / Machine Learning Model Was Used?

The project relies on two core Machine Learning algorithms to compute similarity:

1. **TF-IDF (Term Frequency - Inverse Document Frequency):** 
   - Powered by `sklearn.feature_extraction.text.TfidfVectorizer`.
   - **How it works:** Instead of just counting how often words appear, TF-IDF weighs words so that universally common English words (like "the", "and") carry less weight, while unique, descriptive words (like "space", "dinosaur", "hacker", "superhero") carry higher mathematical significance. This translates our raw texts into a multidimensional numerical vector matrix.

2. **Cosine Similarity:**
   - Powered by `sklearn.metrics.pairwise.cosine_similarity`.
   - **How it works:** Once all movies are translated into numeric vectors, Cosine Similarity measures the angle between those vectors to determine how closely related they are. The system ranks all movies by their similarity score against the searched movie and returns the top 5 highest-ranking matches.

---

## 6. Development AI Assistance
*Note: The refactoring, debugging, and migration of the codebase from legacy Chinese/MySQL scripts to a clean, modern, zero-database, locally-run Scikit-Learn Flask app was directed and executed with the help of **Google Gemini**.*
