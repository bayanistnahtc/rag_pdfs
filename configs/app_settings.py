from enum import Enum
from pydantic import Field
from pydantic_settings import BaseSettings


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Settings(BaseSettings):
    PROJECT_NAME: str = "Rag Medical Assistant"
    API_V1_STR: str = "/api/v1"
     # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")

    # Database settings, API keys, etc.

    class Config:
        env_file = ".env"
        extra = "ignore"
