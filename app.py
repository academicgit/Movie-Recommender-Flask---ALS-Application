import findspark
findspark.init()
from flask import Flask, render_template, request
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
import pandas as pd
import os
import requests
import zipfile
import shutil

app = Flask(__name__)

# URLs and file paths for the MovieLens dataset
dataset_url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
dataset_folder = "ml-latest-small"
ratings_file_path = f"{dataset_folder}/ratings.csv"
movies_file_path = f"{dataset_folder}/movies.csv"

# Function to download and extract the MovieLens dataset
def download_and_extract_dataset():
    if not os.path.exists(dataset_folder):
        print("Downloading dataset...")
        # Download the dataset
        response = requests.get(dataset_url, stream=True)
        with open("ml-latest-small.zip", "wb") as f:
            f.write(response.content)
        
        # Extract the dataset
        print("Extracting dataset...")
        with zipfile.ZipFile("ml-latest-small.zip", "r") as zip_ref:
            zip_ref.extractall(".")
        
        # Clean up the zip file
        os.remove("ml-latest-small.zip")
        print("Dataset ready.")

# Initialize PySpark and Load ALS Model
def initialize_spark_and_model():
    spark = SparkSession.builder.appName('MovieLensRecommendationALS').getOrCreate()
    
    # Load the MovieLens ratings dataset
    ratings = spark.read.csv(ratings_file_path, header=True, inferSchema=True)
    ratings = ratings.select('userId', 'movieId', 'rating')

    # Train ALS model
    als = ALS(maxIter=10, regParam=0.1, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
    model = als.fit(ratings)
    
    return spark, model

# Initialize Spark session and the ALS model
download_and_extract_dataset()  # Ensure the dataset is downloaded and extracted
spark, model = initialize_spark_and_model()

# Load the movies dataset for displaying movie titles
movies = pd.read_csv(movies_file_path)

# Routes for Flask app
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])

    # Get top 5 movie recommendations for the user
    user_recommendations = model.recommendForAllUsers(5).filter(f"userId == {user_id}").collect()

    if len(user_recommendations) == 0:
        return render_template('recommendations.html', movies=None, message=f"No recommendations for user {user_id}.")

    # Extract movie IDs and find corresponding titles
    recommended_movies = [row.movieId for row in user_recommendations[0].recommendations]
    movie_titles = movies[movies['movieId'].isin(recommended_movies)]['title'].tolist()

    return render_template('recommendations.html', movies=movie_titles, message=f"Top 5 movie recommendations for user {user_id}:")

if __name__ == '__main__':
    app.run(debug=True)