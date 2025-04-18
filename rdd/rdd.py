import sys
from pyspark import SparkContext
import time 
import numpy as np
import math
import re

# Initialize Spark Context
sc = SparkContext(appName="Tasks")

# Input paths
input_path = sys.argv[1]
wikiCategoryFile = sys.argv[2]  # reading the categories dataset

# Task 1: Extract content within <doc> tags
rdd = sc.textFile(input_path)
rdd2 = rdd.filter(lambda x: "<doc" in x or "</doc>" in x)

# Task 1.1: Extract document content using regex
def extract_content(doc_line):
    match = re.search(r'<doc.*?>(.*?)</doc>', doc_line, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

rdd3 = rdd2.map(extract_content).filter(lambda x: x is not None)

# Extract words and count the top 20K words
rdd4 = rdd3.flatMap(lambda x: re.findall(r'\b\w+\b', x.lower()))
rdd5 = rdd4.map(lambda x: (x, 1)).reduceByKey(lambda a, b: a + b)
rdd6 = rdd5.takeOrdered(20000, key=lambda x: -x[1])

words_array = [word for word, count in rdd6]
print("Problem 1.1: The top 20K words are:")
print(words_array)

# Task 1.2: Create word index dictionary
word_dict = {word: index for index, word in enumerate(words_array)}

def funcToFindIDandContent(line):
    doc_id_match = re.search(r'id="(\d+)"', line)
    if doc_id_match:
        doc_id = doc_id_match.group(1)
        content_match = re.search(r'>(.*?)</doc>', line, re.DOTALL)
        if content_match:
            content = content_match.group(1).strip()
            return (doc_id, content)
    return None

rdd7 = rdd2.map(funcToFindIDandContent).filter(lambda x: x is not None)

# Task 1.2: Map words in the document to their positions in the top 20K words array
def wordPositions(doc_info):
    doc_id, content = doc_info
    words = re.findall(r'\b\w+\b', content.lower())
    positions = [word_dict[word] for word in words if word in word_dict]
    return (doc_id, np.array(positions))

rdd8 = rdd7.map(wordPositions)
print("Problem 1.2: Doc ID with word positions in the 20K array:")
print(rdd8.take(2))

# Task 2: Calculate TF(w, x)
def TFwx(doc_info):
    doc_id, content = doc_info
    words = re.findall(r'\b\w+\b', content.lower())
    total_words = len(words)
    tf_vector = np.zeros(len(words_array))
    for word in words:
        if word in word_dict:
            index = word_dict[word]
            tf_vector[index] += 1
    if total_words > 0:
        tf_vector = tf_vector / total_words
    return (doc_id, tf_vector)

rdd9 = rdd7.map(TFwx).cache()
print("Problem 2: TF(w, x):")
print(rdd9.take(2))

num_docs = rdd9.count()
print("Total document count:", num_docs)

# Task 2: Calculate IDF
def IDF(doc_info):
    doc_id, tf_vector = doc_info
    return [(i, 1) for i, tf in enumerate(tf_vector) if tf > 0]

df_rdd = rdd9.flatMap(IDF).reduceByKey(lambda a, b: a + b)
idf_rdd = df_rdd.mapValues(lambda df: math.log(num_docs / (1 + df)))
idf_dict = idf_rdd.collectAsMap()
idf_broadcast = sc.broadcast(idf_dict)

print("10000th word IDF:", idf_dict.get(10000, "Not in top 20K"))

# Task 2: Calculate TF-IDF
def TFIDF(doc_info):
    doc_id, tf_vector = doc_info
    tf_idf_vector = np.zeros_like(tf_vector)
    for i in range(len(tf_vector)):
        idf_value = idf_broadcast.value.get(i, 0)
        tf_idf_vector[i] = tf_vector[i] * idf_value
    return (doc_id, tf_idf_vector)

rdd10 = rdd9.map(TFIDF)
print("TF-IDF for each word in a document:")
print(rdd10.take(2))

# Task 3: Load and process categories data
wikiCategoryLinks = sc.textFile(wikiCategoryFile)
wikiCats = wikiCategoryLinks.map(lambda x: x.split(",")).map(lambda x: (x[0].replace('"', ''), x[1].replace('"', '')))
print("Problem 3: Reading the categories:")
print(wikiCats.take(5))

# Task 3: Cosine similarity function
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

# Task 3: Predict top k documents based on cosine similarity
def getPrediction(textInput, k):
    words = textInput.split()
    input_vector = np.zeros(len(words_array))
    
    for word in words:
        if word in words_array:
            index = word_dict[word]
            input_vector[index] += 1
    input_vector = input_vector / np.linalg.norm(input_vector)
    
    # Use mapPartitions to avoid collect() and process in chunks
    def compute_similarity(partition):
        similarities = []
        for doc_id, tf_vector in partition:
            similarity = cosine_similarity(input_vector, tf_vector)
            similarities.append((doc_id, similarity))
        return similarities
    
    similarities_rdd = rdd9.mapPartitions(compute_similarity)
    top_documents = similarities_rdd.takeOrdered(k, key=lambda x: -x[1])
    top_doc_ids_with_similarities = [(doc[0], doc[1]) for doc in top_documents]
    
    return top_doc_ids_with_similarities

# Example prediction
print("Example 1: What is quantitative finance?")
textInput = "What is quantitative finance?"
k = 20
predicted_docs_with_similarities = getPrediction(textInput, k)

print("Predicted doc IDs with similarities:", predicted_docs_with_similarities)

# Broadcast predicted doc IDs with their similarities
predicted_doc_broadcast = sc.broadcast(predicted_docs_with_similarities)

# Task 3: Include similarity score and filter categories
filtered_wikiCats_with_similarity = wikiCats.map(lambda x: (x[0], x[1], dict(predicted_doc_broadcast.value).get(x[0], None))) \
                                            .filter(lambda x: x[2] is not None)

result_with_similarity = filtered_wikiCats_with_similarity.collect()

print("Predicted doc IDs, categories, and similarities:")
print(result_with_similarity)

# Stop SparkContext
sc.stop()
