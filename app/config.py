from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b-instruct"
    ollama_embed_model: str = "qwen3-embedding:0.6b"
    max_tool_loops: int = 6
    arxiv_max_results: int = 10
    request_timeout_seconds: int = 120
    data_dir: Path = PROJECT_ROOT / "data"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "research.db"

    @property
    def papers_dir(self) -> Path:
        return self.data_dir / "papers"

    @property
    def indexes_dir(self) -> Path:
        return self.data_dir / "indexes"

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.papers_dir, self.indexes_dir):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()

