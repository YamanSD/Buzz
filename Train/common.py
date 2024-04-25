from collections import Counter

from joblib import dump, load as jload
from os import path

from numpy import array, ndarray, flip, zeros, argsort
from sklearn.feature_extraction.text import CountVectorizer
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
        document_term_matrix: ndarray,
        pipeline: Pipeline
) -> list[str]:
    """
    returns a list of n_topic strings, where each string contains the n most common
    words in a predicted category, in order
    """
    top_word_indices = []
    for topic in range(n_topics):
        temp_vector_sum = 0
        for i in range(len(keys)):
            if keys[i] == topic:
                temp_vector_sum += document_term_matrix[i]
        temp_vector_sum = temp_vector_sum.toarray()
        top_n_word_indices = flip(argsort(temp_vector_sum)[0][-n:], 0)
        top_word_indices.append(top_n_word_indices)
    top_words = []
    count_vectorizer: CountVectorizer = pipeline.named_steps['vect']

    for topic in top_word_indices:
        topic_words = []
        for index in topic:
            temp_word_vector = zeros((1, document_term_matrix.shape[1]))
            temp_word_vector[:, index] = 1
            the_word = count_vectorizer.inverse_transform(temp_word_vector)[0][0]
            topic_words.append(the_word.encode('ascii').decode('utf-8'))
        top_words.append(" ".join(topic_words))
    return top_words
