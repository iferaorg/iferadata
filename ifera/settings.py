"""
Settings module for the ifera package.
Contains configuration settings that can be loaded from environment variables or .env file.
"""

from pydantic import ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict


# pylint: disable=too-few-public-methods
class Settings(BaseSettings):
    """
    Application settings class that handles configuration through environment variables.
    Default values are provided but can be overridden through environment variables or .env file.
    """

    DATA_FOLDER: str = "data"
    S3_BUCKET: str = "s3bucket"
    GITHUB_TOKEN: str = ""

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
