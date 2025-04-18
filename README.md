# TF-IDF and k-NN Text Classification with PySpark

This mini-project implements a text classification pipeline using PySpark. The project demonstrates how to process Wikipedia-based document data to generate a dictionary of top 20,000 words, represent documents as TF-IDF vectors, and classify new input using k-Nearest Neighbors (k-NN) with cosine similarity.

## 📁 Project Structure

- `rdd/`: Implements the pipeline using PySpark RDDs.
- `dataframe/`: Implements the same pipeline using PySpark DataFrames.
- `output/`: Contains sample outputs from both implementations.

## 🚀 Functional Highlights

- **Top 20,000 Dictionary Extraction** from Wikipedia corpus.
- **TF-IDF vectorization** using Spark.
- **Cosine similarity based kNN classifier**.
- **Dual implementations**: RDD and DataFrame.

## 🧠 Functions Implemented

- `getTop20KWords()`: Extracts top 20,000 frequent words.
- `generateTFIDF()`: Creates TF-IDF vectors for documents.
- `getPrediction(textInput, k)`: Predicts category for input using cosine similarity.
