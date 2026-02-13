"""Configuration management for Discord AI Bot."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    discord_token: str
    google_credentials_path: Path
    ollama_host: str
    ollama_model: str
    evaluation_prompt: str

    @classmethod
    def load(cls) -> "Config":
        """Load configuration from environment variables.

        Raises:
            ValueError: If required environment variables are missing.
        """
        load_dotenv()

        discord_token = os.getenv("DISCORD_TOKEN")
        if not discord_token:
            raise ValueError("DISCORD_TOKEN environment variable is required")

        google_credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH", "./credentials.json")

        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "minicpm-v")

        evaluation_prompt = os.getenv(
            "EVALUATION_PROMPT",
            "この画像を評価し、内容の説明と改善点を日本語で回答してください。"
        )

        return cls(
            discord_token=discord_token,
            google_credentials_path=Path(google_credentials_path),
            ollama_host=ollama_host,
            ollama_model=ollama_model,
            evaluation_prompt=evaluation_prompt,
        )
