from __future__ import annotations

from dataclasses import dataclass, asdict
from requests import post
from typing import Optional, Iterable

from Config import config


@dataclass
class SentimentResponse:
    """
    Class used for sentiment queries.


    positive: SentimentResponse of the positive sentiment.

    negative: SentimentResponse of the negative sentiment.

    neutral: SentimentResponse of the neutral sentiment.
    """
    positive: float
    negative: float
    neutral: float

    @classmethod
    def fromlist(cls, sentiments: list[dict]) -> SentimentResponse:
        temp: dict = {
            s['label']: s['score'] for s in sentiments
        }

        return cls(
            temp['positive'],
            temp['negative'],
            temp['neutral']
        )

    def __add__(self, other: SentimentResponse) -> SentimentResponse:
        return SentimentResponse(
            self.positive + other.positive,
            self.negative + other.negative,
            self.neutral + other.neutral
        )

    def __truediv__(self, other: float) -> SentimentResponse:
        return SentimentResponse(
            self.positive / other,
            self.negative / other,
            self.neutral / other
        )

    def __mul__(self, other: float) -> SentimentResponse:
        return SentimentResponse(
            self.positive * other,
            self.negative * other,
            self.neutral * other
        )

    def __iadd__(self, other: SentimentResponse) -> None:
        self.positive += other.positive
        self.negative += other.negative
        self.neutral += other.neutral

    def __imul__(self, other: float) -> None:
        self.positive *= other
        self.negative *= other
        self.neutral *= other

    def __itruediv__(self, other: float) -> None:
        self.positive /= other
        self.negative /= other
        self.neutral /= other

    def net_sentiment(self) -> float:
        return self.positive - self.negative


@dataclass(frozen=True)
class SentimentRequestOptions:
    """
    Class used for sentiment query options.


    use_cache: True to use caching.

    wait_for_model: True to wait for the model if not booted (due to serverless cold starts).
    """
    use_cache: Optional[bool]
    wait_for_model: Optional[bool]


@dataclass(frozen=True)
class SentimentRequest:
    """
    Class used for sentiment queries.


    inputs: list of queries to the model or a single string query.

    options: HF request options.
    """
    inputs: str | Iterable[str]
    options: Optional[SentimentRequestOptions] = SentimentRequestOptions(True, True)


def query(payload: SentimentRequest) -> list[SentimentResponse]:
    """

    Args:
        payload: Necessary data to perform the sentiment request.

    Returns:
        The sentiment of the given inputs according to an NLP model.

    """

    # For documentation of requests consult: https://docs.python-requests.org/en/latest/user/advanced/
    # For documentation of API check config.json for API link
    return list(
        map(
            lambda ms: SentimentResponse.fromlist(ms),
            post(
                config.hf.sentiment_url,
                headers={
                    "Authorization": config.hf.sentiment_token
                },
                json=asdict(payload)
            ).json()
        )
    )


def relative_sentiment(news: str) -> SentimentResponse:
    """

    Args:
        news: News title to analyze.

    Returns:
        The weighted average sentiment response based on the given weight function.

    """

    # Query the NLP model
    return query(SentimentRequest(news))[0]
