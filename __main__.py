import nltk
from multiprocessing import Process, Queue
from flask import Flask, request, render_template, flash, redirect, url_for
from newspaper import Article, ArticleException
from requests import get, RequestException
from re import split
from sklearn.pipeline import Pipeline
from textblob import TextBlob
from urllib.parse import urlparse
from validators import url as v_url

from Config import config
from Sentiment import relative_sentiment
from Train import load_lsa, load_lda, extract_topic


app: Flask = Flask(__name__)

# Load the models
lsa_model: Pipeline = load_lsa()
lda_model: Pipeline = load_lda()

# Model queues for multiprocessing
lda_q: Queue = Queue()
lsa_q: Queue = Queue()


def get_website_name(url: str) -> str:
    """

    Args:
        url: URl to extract the website name from.

    Returns:
        The extracted website name.

    """
    domain: str = urlparse(url).netloc

    if domain.startswith("www."):
        domain = domain[4:]

    return domain


def run(
    title: str,
    model: Pipeline,
    q: Queue
) -> None:
    """

    Args:
        title: To feed to the model.
        model: Model to use.
        q: Process queue, contains the final string.

    """
    q.put_nowait(', '.join(extract_topic(model, 5, title)).upper())


@app.route('/', methods=['GET', 'POST'])
def index():
    global lsa_model, lda_model, lsa_q, lda_q

    if request.method == 'POST':
        url: str = request.form['url']

        # Check if the input is a valid URL
        if not v_url(url):
            flash('Please enter a valid URL.')
            return redirect(url_for('index'))

        try:
            # Raise an HTTPError if the HTTP request returned an unsuccessful status code
            get(url).raise_for_status()

            # Article might fail to download
            article: Article = Article(url)
            article.download()
        except (RequestException, ArticleException):
            flash('Failed to download the content of the URL.')
            return redirect(url_for('index'))

        article.parse()

        # Perform natural language processing
        article.nlp()

        title: str = article.title
        authors: str = ', '.join(article.authors)

        if not authors:
            authors = get_website_name(url)  # Set the author field to the website name

        publish_date: str = article.publish_date.strftime('%B %d, %Y') if article.publish_date else "N/A"

        # Manually adjust the summary length by selecting a certain number of sentences
        sentences: list[str] = split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.?])\s", article.text)

        # Adjust the number of sentences as needed
        max_summarized_sentences: int = 5

        summary: str = ' '.join(sentences[:max_summarized_sentences])

        if not len(summary):
            flash('Please enter a valid URL.')
            return redirect(url_for('index'))

        # Start LSA model
        Process(target=run, args=(summary, lsa_model, lsa_q)).start()

        # Start LDA model
        Process(target=run, args=(summary, lda_model, lda_q)).start()

        # Get the top image URL
        top_image: str = article.top_image

        analysis: TextBlob = TextBlob(article.text)

        # Analyze the financial sentiment of the article
        net_sentiment: float = relative_sentiment(summary).net_sentiment()
        sentiment: str = "Neutral ⬛"

        if net_sentiment > 0:
            sentiment = "Positive 🟩"
        elif net_sentiment < 0:
            sentiment = "Negative 🟥"

        return render_template(
            'index.html',
            title=title,
            authors=authors,
            publish_date=publish_date,
            summary=summary,
            top_image=top_image,
            subjectivity=f"{analysis.sentiment.subjectivity:.2%}",
            sentiment=sentiment,
            lsa_topic=lsa_q.get(),
            lda_topic=lda_q.get(),
        )

    return render_template('index.html')


app.secret_key = config.server.secret

if __name__ == '__main__':
    # Download punctuation extension
    nltk.download('punkt')

    app.run(debug=True, port=config.server.port, use_reloader=False)
