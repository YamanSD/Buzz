from collections import Counter

from joblib import dump, load as jload
from os import path

from numpy import mean, ndarray, flip, zeros, argsort, vstack
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Directory path
dir_path: str = path.dirname(path.realpath(__file__))


def load(file_name: str) -> Pipeline:
    return jload(path.join(dir_path, f"{file_name}.sav"))


def save(model: Pipeline, file_name: str) -> None:
    dump(model, path.join(dir_path, f"{file_name}.sav"))


def get_keys(topic_matrix: ndarray) -> ndarray:
    """
    returns an integer list of predicted topic
    categories for a given topic matrix
    """
    return topic_matrix.argmax(axis=1).tolist()


def keys_to_counts(keys: ndarray) -> tuple[list[int], list[int]]:
    """
    returns a tuple of topic categories and their
    accompanying magnitudes for a given list of keys
    """
    count_pairs = Counter(keys).items()
    categories = [pair[0] for pair in count_pairs]
    counts = [pair[1] for pair in count_pairs]

    return categories, counts


def get_top_n_words(
        n_topics: int,
        n: int,
        keys: ndarray,
        document_term_matrix: csr_matrix | ndarray,
        pipeline: Pipeline | TfidfVectorizer
) -> list[str]:
    """
    returns a list of n_topic strings, where each string contains the n most common
    words in a predicted category, in order
    """
    top_word_indices = []
    for topic in range(n_topics):
        temp_vector_sum: csr_matrix | int | ndarray = 0

        for i in range(len(keys)):
            if keys[i] == topic:
                # This transforms temp_vector_sum to a csr_matrix
                temp_vector_sum += document_term_matrix[i]

        if type(temp_vector_sum) is int:
            continue

        temp_vector_sum = temp_vector_sum.toarray()
        top_n_word_indices = flip(argsort(temp_vector_sum)[0][-n:], 0)
        top_word_indices.append(top_n_word_indices)
    top_words = []
    vectorizer: TfidfVectorizer = pipeline.named_steps['vect'] if type(pipeline) is Pipeline else pipeline

    for topic in top_word_indices:
        topic_words = []
        for index in topic:
            temp_word_vector = zeros((1, document_term_matrix.shape[1]))
            temp_word_vector[:, index] = 1
            the_word = vectorizer.inverse_transform(temp_word_vector)[0][0]
            topic_words.append(the_word.encode('ascii').decode('utf-8'))
        top_words.append(" ".join(topic_words))
    return top_words


def get_mean_topic_vectors(n_topics: int, keys: ndarray, two_dim_vectors: ndarray) -> list[list[float]]:
    """
    returns a list of centroid vectors from each predicted topic category
    """
    mean_topic_vectors = []
    for t in range(n_topics):
        articles_in_that_topic = []
        for i in range(len(keys)):
            if keys[i] == t:
                articles_in_that_topic.append(two_dim_vectors[i])

        articles_in_that_topic = vstack(articles_in_that_topic)
        mean_article_in_that_topic = mean(articles_in_that_topic, axis=0)
        mean_topic_vectors.append(mean_article_in_that_topic)
    return mean_topic_vectors
