from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATA_FOLDER: str = "data"
    S3_BUCKET: str = "rawdata"
    S3_BUCKET_PROCESSED: str = "processeddata"

    class Config:
        env_file = ".env"

settings = Settings()
