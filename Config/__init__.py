from .config import load_config, Config, HfConfig, ProxiesConfig, KaggleConfig, ServerConfig, TrainingConfig

# Default config instance
config: Config = load_config("./config.json")
