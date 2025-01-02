import requests
import zipfile
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import logging
import csv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MusicRecommendationModel:
    def __init__(self, download_url, download_path, extract_path):
        # Initialize Spark session
        self.spark = SparkSession.builder \
            .appName("MusicRecommendationModel") \
            .getOrCreate()

        # Dataset URL and file paths
        self.download_url = download_url
        self.download_path = download_path
        self.extract_path = extract_path

    def download_and_extract_data(self):
        """
        Downloads and extracts the dataset from the provided URL.
        """
        # Download the dataset
        logger.info(f"Downloading dataset from {self.download_url}...")
        response = requests.get(self.download_url)
        
        # Write the zip file to disk
        with open(self.download_path, 'wb') as file:
            file.write(response.content)
        logger.info(f"Dataset downloaded to {self.download_path}")
        
        # Extract the downloaded zip file
        logger.info(f"Extracting dataset to {self.extract_path}...")
        with zipfile.ZipFile(self.download_path, 'r') as zip_ref:
            zip_ref.extractall(self.extract_path)
        logger.info(f"Dataset extracted to {self.extract_path}")

    def load_data(self):
        """
        Loads and preprocesses the dataset.
        """
        logger.info("Loading dataset...")

        # Load ratings data (columns: user_id, song_id, rating)
        ratings_data = self.spark.read.csv(f"{self.extract_path}/ml-100k/u.data", sep="\t", header=False, inferSchema=True)
        ratings_data = ratings_data.withColumnRenamed("_c0", "user_id") \
                                   .withColumnRenamed("_c1", "song_id") \
                                   .withColumnRenamed("_c2", "rating")
        logger.info("Ratings data loaded successfully.")

        # Load song metadata (columns: song_id, song_name)
        song_metadata = self.spark.read.csv(f"{self.extract_path}/ml-100k/u.item", sep="|", header=False, inferSchema=True)
        song_metadata = song_metadata.withColumnRenamed("_c0", "song_id") \
                                       .withColumnRenamed("_c1", "song_name")
        logger.info("Song metadata loaded successfully.")

        # Join ratings with metadata to get song names
        self.ratings_df = ratings_data.join(song_metadata, on="song_id", how="inner")
        logger.info(f"Joined ratings with song metadata. Total records: {self.ratings_df.count()}")

    def train_model(self):
        """
        Trains a collaborative filtering model using ALS (Alternating Least Squares).
        """
        logger.info("Training the ALS model...")

        # Initialize ALS model
        als = ALS(userCol="user_id", itemCol="song_id", ratingCol="rating", coldStartStrategy="drop")

        # Split the data into training and testing sets
        training_data, test_data = self.ratings_df.randomSplit([0.8, 0.2], seed=42)
        logger.info(f"Training set: {training_data.count()} records, Testing set: {test_data.count()} records")

        # Train the ALS model
        model = als.fit(training_data)

        # Make predictions on the test data
        predictions = model.transform(test_data)

        # Evaluate the model
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        rmse = evaluator.evaluate(predictions)
        logger.info(f"Root-Mean-Square Error (RMSE) on test data = {rmse}")

        # Return the trained model
        return model

    def recommend_songs(self, num_recommendations=5):
        """
        Recommends songs for all users based on the trained model and saves the recommendations in CSV using the csv module.
        """
        logger.info(f"Making {num_recommendations} recommendations for all users...")

        # Load the trained model
        model = self.train_model()

        # Generate recommendations for all users
        user_recs = model.recommendForAllUsers(num_recommendations)

        # Flatten recommendations to get song names and ratings
        user_recs_flat = user_recs.select(
            "user_id",
            explode("recommendations").alias("recommendation")
        ).select(
            "user_id", 
            "recommendation.song_id", 
            "recommendation.rating"
        )

        # Join with song metadata to get song names
        recommended_songs = user_recs_flat.join(
            self.ratings_df, 
            user_recs_flat.song_id == self.ratings_df.song_id, 
            how="inner"
        ).select(
            user_recs_flat.user_id,
            self.ratings_df.song_name,
            user_recs_flat.rating
        )

        # Remove duplicates, order by user_id
        ordered_songs = recommended_songs.distinct().orderBy("user_id")

        # Collect data for writing
        ordered_songs_collected = ordered_songs.collect()

        # Define the CSV file path
        csv_file_path = "ordered_recommended_songs.csv"

        # Write to a single CSV file using Python's csv module
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow(["user_id", "song_name", "rating"])

            # Write the data for all users in order
            for row in ordered_songs_collected:
                writer.writerow([row["user_id"], row["song_name"], row["rating"]])

        logger.info(f"Ordered user recommendations saved to {csv_file_path}.")
        return ordered_songs

    def run(self):
        """
        Main function to run the entire recommendation process.
        """
        logger.info("Starting the music recommendation model...")

        # Download and extract the dataset
        self.download_and_extract_data()

        # Load data into Spark
        self.load_data()

        # Train the model and evaluate
        self.train_model()

        # Get recommendations
        self.recommend_songs(num_recommendations=5)


# Define the dataset download URL and file paths
download_url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
download_path = "ml-100k.zip"
extract_path = "ml-100k"

# Instantiate and run the music recommendation model
music_rec_model = MusicRecommendationModel(download_url, download_path, extract_path)
music_rec_model.run()
