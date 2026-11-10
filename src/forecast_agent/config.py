from pathlib import Path

from pydantic import BaseModel, ConfigDict


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    database_path: Path = Path("data/forecast-agent.sqlite3")
