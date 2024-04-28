from numpy import array, ndarray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from .common import get_keys, get_top_n_words
from .lsa import train as train_lsa, load as load_lsa, test as test_lsa
from .lda import train as train_lda, load as load_lda, test as test_lda


def extract_topic(model: Pipeline, title: str) -> str:
    sample: ndarray = array([title.lower()])

    vect: TfidfVectorizer = model.named_steps['vect']

    term_mat: ndarray = vect.transform(sample)
    topic_mat: ndarray = model.transform(sample)

    keys: ndarray = get_keys(topic_mat)

    return ', '.join(get_top_n_words(8, 3, keys, term_mat, vect))

