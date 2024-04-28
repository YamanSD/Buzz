from bokeh.io import show
from bokeh.models import Label
from bokeh.plotting import figure
from numpy import ndarray, array
from pandas import DataFrame
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline

from Data import get_data

from .common import load as g_load, save as g_save, get_keys, get_top_n_words, get_mean_topic_vectors


# Name of the model file, without extension
save_file: str = "lsa_model"

# Number of iterations for the model
iters: int = 16


def test(max_features: int, n_topics: int, n_records: int) -> None:
    """

    Args:
        max_features: Number of max features to test on.
        n_topics: Number of topics to test on.
        n_records: Number of records to test on.

    """
    global iters

    # Part of the data set
    data: DataFrame = get_data().sample(n_records)

    # Vectorizer
    vect: TfidfVectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=max_features
    )

    # Get the samples
    text_sample: ndarray = data.values.ravel()
    term_matrix: ndarray = vect.fit_transform(text_sample)

    # LSA model
    model: TruncatedSVD = TruncatedSVD(n_components=n_topics, n_iter=iters)

    topic_matrix: ndarray = model.fit_transform(term_matrix)

    keys: ndarray = get_keys(topic_matrix)

    tsne_model: TSNE = TSNE(
        n_components=2,
        perplexity=50,
        learning_rate=100,
        n_iter=2000,
        verbose=1,
        random_state=0,
        angle=0.75
    )
    tsne_lsa_vectors: ndarray = tsne_model.fit_transform(topic_matrix)

    # Colormap for the plot
    colormap = array([
        "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
        "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
        "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
        "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"])[:n_topics]

    top_words: list[str] = get_top_n_words(n_topics, 3, keys, term_matrix, vect)

    mean_topic_vectors = get_mean_topic_vectors(n_topics, keys, tsne_lsa_vectors)

    plot = figure(title="t-SNE Clustering of {} LSA Topics".format(n_topics))
    plot.scatter(x=tsne_lsa_vectors[:, 0], y=tsne_lsa_vectors[:, 1], color=colormap[keys])

    for t in range(n_topics):
        label = Label(x=mean_topic_vectors[t][0], y=mean_topic_vectors[t][1],
                      text=top_words[t], text_color=colormap[t])
        plot.add_layout(label)

    show(plot)


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
        ('vect', TfidfVectorizer(stop_words='english', max_features=max_features)),
        ('model', TruncatedSVD(n_components=n_topics, n_iter=iters)),
    ])

    # LSA model
    topic_matrix: ndarray = pipeline.fit_transform(data_values)

    return pipeline, topic_matrix, pipeline.named_steps['vect'].transform(data_values)


def train(max_features: int, n_topics: int, no_save: bool = False) -> tuple[Pipeline, ndarray, ndarray]:
    """

    Trains the model and saves it to its designated file.

    Args:
        max_features: The top most frequent words to keep.
        n_topics: Number of unique topics.
        no_save: True to not save the trained model.

    Returns:
        The trained model along with the topic matrix and the term matrix.

    """
    data: DataFrame = get_data()

    # Train the model
    pipeline, topic_matrix, term_matrix = simple_train(
        max_features,
        n_topics,
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
