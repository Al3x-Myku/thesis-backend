# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # only these two will be picked up
    SECRET_KEY: str
    DATABASE_URL: str

    # you still get sensible defaults for your JWT logic
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 1 week

    # load from project-root .env and ignore everything else in it
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

# singleton instance
settings = Settings()
