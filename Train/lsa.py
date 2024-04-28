from re import sub

from numpy import ndarray
from pandas import DataFrame
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from Config import config
from Data import get_data

from .common import load as g_load, save as g_save, apply_tsne


# Name of the model file, without extension
save_file: str = "lsa_model"

# Number of iterations for the model
iters: int = 16


def test(n_records: int) -> None:
    """

    Args:
        n_records: Number of records to test on.

    """
    global iters

    # Part of the data set
    data: DataFrame = get_data().sample(n_records)

    # Number of topics from config
    n_topics: int = config.training.n_topics

    # Vectorizer
    vect: TfidfVectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=config.training.max_features,
    )

    # Get the samples
    text_sample: ndarray = data.values.ravel()
    term_matrix: ndarray = vect.fit_transform(text_sample)

    # LSA model
    model: TruncatedSVD = TruncatedSVD(n_components=n_topics, n_iter=iters)

    apply_tsne(n_topics, model, vect, term_matrix)


def simple_train(max_features: int, n_topics: int, data: DataFrame) -> tuple[Pipeline, ndarray, ndarray]:
    """

    Args:
        max_features: The top most frequent words to keep.
        n_topics: number of unique topics.
        data: training data.

    Returns:
        The model pipline along with the topics matrix and the term matrix.

    """
    global iters

    # Extract the data values array
    data_values: ndarray = data.values.ravel()

    # Vectorize the input, then pass it to the model, & apply t-sne dimensionality reduction
    pipeline: Pipeline = Pipeline([
        ('vect', TfidfVectorizer(
            stop_words='english',
            max_features=max_features,
        )),
        ('model', TruncatedSVD(n_components=n_topics, n_iter=iters)),
    ])

    # LSA model
    topic_matrix: ndarray = pipeline.fit_transform(data_values)

    return pipeline, topic_matrix, pipeline.named_steps['vect'].transform(data_values)


def train(no_save: bool = False) -> tuple[Pipeline, ndarray, ndarray]:
    """

    Trains the model and saves it to its designated file.

    Args:
        no_save: True to not save the trained model.

    Returns:
        The trained model along with the topic matrix and the term matrix.

    """
    data: DataFrame = get_data()

    # Train the model
    pipeline, topic_matrix, term_matrix = simple_train(
        config.training.max_features,
        config.training.n_topics,
        data
    )

    # Evaluate the model
    if not no_save:
        save(pipeline)

    return pipeline, topic_matrix, term_matrix


def load() -> Pipeline:
    """

    Returns:
        The loaded model from the designated file.

    """
    return g_load(save_file)


def save(model: Pipeline) -> None:
    """

    Saves the model into its file.

    Args:
        model: Model to be saved into the file.


    """
    g_save(model, save_file)
