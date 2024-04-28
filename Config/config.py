from dataclasses import dataclass
from typing import TypedDict


from Utils import convert_to_dataclass, read_json


@dataclass(frozen=True)
class HfConfig:
    """
    Class used for HuggingFace configuration.


    sentiment_token: Bearer token for the sentiment query requests.

    sentiment_url: URL to the sentiment model API.
    """
    sentiment_token: str
    sentiment_url: str


@dataclass(frozen=True)
class KaggleConfig:
    """
    Class used for Kaggle configuration.


    username: Kaggle username.

    key: Kaggle auth key.
    """
    username: str
    key: str


class ProxiesConfig(TypedDict):
    """
    Class used for proxy configuration.


    http: HTTP proxy.

    https: HTTPS proxy.

    https_y: HTTPS proxy used by Yahoo.
    """
    http: str
    https: str
    https_y: str


@dataclass(frozen=True)
class ServerConfig:
    """
    Class used for web server configuration.


    port: Port number of the web server.
    secret: Server secret for encryption.
    """
    port: int
    secret: str


@dataclass(frozen=True)
class TrainingConfig:
    """
    Class used for training configuration.


    n_topics: Number of unique topics.
    max_features: Maximum number of words to consider by the vectorizer.
    """
    n_topics: int
    max_features: int


@dataclass(frozen=True)
class Config:
    """
    Class used for app configuration.
    """
    hf: HfConfig
    proxies: ProxiesConfig
    kaggle: KaggleConfig
    server: ServerConfig
    training: TrainingConfig


def load_config(path: str) -> Config:
    """

    Reads the given JSON config file.

    Args:
        path: Path to the config file.

    Returns:
        The loaded config object.

    """
    data: dict = read_json(path)

    # Convert to ConfigType object and return
    return convert_to_dataclass(Config, {
        **data,
        "hf": convert_to_dataclass(HfConfig, data['hf']),
        "kaggle": convert_to_dataclass(KaggleConfig, data['kaggle']),
        "server": convert_to_dataclass(ServerConfig, data['server']),
        "training": convert_to_dataclass(TrainingConfig, data['training']),
    })
