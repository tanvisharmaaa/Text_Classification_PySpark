import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, explode, trim, udf, regexp_extract
from pyspark.ml.linalg import SparseVector, VectorUDT
from pyspark.ml.feature import Normalizer
from pyspark.sql.types import DoubleType
import numpy as np

# Initialize Spark session
spark = SparkSession.builder.appName("KNN_Text_Classification_Optimized").getOrCreate()

# Use sys.argv for input files
wiki_pages_path = sys.argv[1]  # Pass 'WikipediaPagesOneDocPerLine1000LinesSmall.txt' as argument
category_path = sys.argv[2]    # Pass 'wiki-categorylinks-small.csv.bz2' as argument

# Load the Wikipedia Pages file
df_wiki_pages = spark.read.text(wiki_pages_path)

# Extract docID and content from the Wikipedia pages
df_wiki_pages = df_wiki_pages.withColumn('doc_id', regexp_extract(col('value'), r'id="(\d+)"', 1)) \
                             .withColumn('content', regexp_extract(col('value'), r'>(.*?)</doc>', 1)) \
                             .filter(col('doc_id') != "").filter(col('content') != "")

# Load and process the category file (wiki-categorylinks-small.csv.bz2)
categories_rdd = spark.sparkContext.textFile(category_path)

# Apply the logic to split by commas and remove quotes from doc_id and category
categories_rdd = categories_rdd.map(lambda x: x.split(",")).map(lambda x: (x[0].replace('"', ''), x[1].replace('"', '')))
categories_df = categories_rdd.toDF(["doc_id", "category"]).withColumn('doc_id', trim(col('doc_id')))

# Dictionary creation from the Wikipedia content (top 20K words)
df_words = df_wiki_pages.withColumn('words', explode(split(col('content'), r'\s+')))
df_word_counts = df_words.groupBy('words').count().orderBy(col('count').desc()).limit(20000)

# Create a dictionary of words and their corresponding indices
word_dict = {row['words']: index for index, row in enumerate(df_word_counts.collect())}

# Broadcast the word_dict to all worker nodes
word_dict_broadcast = spark.sparkContext.broadcast(word_dict)

# Task 2: Compute Term Frequencies (TF) using UDF and DataFrames
def calculate_tf(content):
    word_dict = word_dict_broadcast.value  # Access the broadcast variable
    words = content.lower().split()
    total_words = len(words)
    word_counts = {}

    for word in words:
        if word in word_dict:
            index = word_dict[word]
            word_counts[index] = word_counts.get(index, 0) + 1

    indices = list(word_counts.keys())
    values = [word_counts[idx] / total_words for idx in indices]

    # Sort indices and values together to ensure strictly increasing order of indices
    sorted_indices_and_values = sorted(zip(indices, values), key=lambda x: x[0])
    sorted_indices = [idx for idx, _ in sorted_indices_and_values]
    sorted_values = [val for _, val in sorted_indices_and_values]

    return SparseVector(len(word_dict), sorted_indices, sorted_values)

# Apply the TF calculation as a UDF
tf_udf = udf(calculate_tf, VectorUDT())
df_wiki_pages = df_wiki_pages.withColumn('tf_vector', tf_udf(col('content')))

# Normalize TF vectors
normalizer = Normalizer(inputCol="tf_vector", outputCol="norm_tfidf")
df_normalized = normalizer.transform(df_wiki_pages)

# Task 3: Function to calculate similarity using a UDF
def cosine_similarity(doc_vector, input_vector):
    dot_product = float(np.dot(doc_vector.toArray(), input_vector.toArray()))
    norm1 = float(np.linalg.norm(doc_vector.toArray()))
    norm2 = float(np.linalg.norm(input_vector.toArray()))

    # Check if either norm is zero to avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return 0.0  # Return zero similarity if any vector is zero

    return dot_product / (norm1 * norm2)

# Main function to get top K predictions
def getPrediction(text_input, k, df_normalized):
    # Preprocess input text
    input_words = text_input.lower().split()
    input_vector = np.zeros(len(word_dict_broadcast.value))

    for word in input_words:
        if word in word_dict_broadcast.value:
            index = word_dict_broadcast.value[word]
            input_vector[index] += 1

    indices = [i for i in range(len(input_vector)) if input_vector[i] > 0]
    values = [input_vector[i] for i in indices]
    input_vector_sparse = SparseVector(len(word_dict_broadcast.value), indices, values)

    # Broadcast the input vector
    input_vector_broadcast = spark.sparkContext.broadcast(input_vector_sparse)

    # Calculate cosine similarity using UDF
    cosine_udf = udf(lambda doc_vector: cosine_similarity(doc_vector, input_vector_broadcast.value), DoubleType())
    df_with_similarity = df_normalized.withColumn("cosine_similarity", cosine_udf(col("norm_tfidf")))

    # Get top K documents by similarity
    top_k_docs = df_with_similarity.orderBy(col("cosine_similarity").desc()).limit(k).select("doc_id", "cosine_similarity")

    # Join with categories and return the result as a DataFrame
    top_k_results = top_k_docs.join(categories_df, "doc_id").select("doc_id", "cosine_similarity", "category")
    return top_k_results

# Example usage of getPrediction
text_input = "What is machine learning?"
k = 5
predictions_df = getPrediction(text_input, k, df_normalized)

# Display top K results as a Spark DataFrame
predictions_df.show(truncate=False)
