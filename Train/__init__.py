from numpy import array, ndarray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from .common import get_keys, extract_top_words
from .lsa import train as train_lsa, load as load_lsa, test as test_lsa
from .lda import train as train_lda, load as load_lda, test as test_lda


def extract_topic(model: Pipeline, n: int, title: str) -> list[str]:
    """

    Args:
        model: Model to use.
        n:  Top n-words to return.
        title: String from which to extract the topic.

    Returns:
        List of most popular words for the topic in decreasing order of popularity.

    """
    sample: ndarray = array([title.lower()])
    vect: TfidfVectorizer = model.named_steps['vect']

    return extract_top_words(
        n,
        get_keys(model.transform(sample)),
        vect.transform(sample),
        vect
    )
