from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ForecastConfig(StrictModel):
    horizon_days: Literal[7]
    input_windows: tuple[Literal[28, 56], ...]


class ScenarioConfig(StrictModel):
    start_date: date
    end_date: date
    seed: int
    item_count: Literal[15]
    ingredient_count: Literal[15]

    @model_validator(mode="after")
    def require_730_days(self) -> "ScenarioConfig":
        if (self.end_date - self.start_date).days + 1 != 730:
            raise ValueError("scenario must contain exactly 730 days")
        if self.start_date != date(2024, 1, 2) or self.end_date != date(2025, 12, 31):
            raise ValueError("scenario dates must be 2024-01-02 through 2025-12-31")
        return self


class EvaluationConfig(StrictModel):
    validation_folds: Literal[8] = 8
    test_days: Literal[28] = 28


class LocationConfig(StrictModel):
    latitude: float = 10.8231
    longitude: float = 106.6297
    timezone: Literal["Asia/Ho_Chi_Minh"] = "Asia/Ho_Chi_Minh"


class PathsConfig(StrictModel):
    data_root: Path = Path("data")
    artifact_root: Path = Path("artifacts")
    cache_root: Path = Path(".cache")


class ProjectConfig(StrictModel):
    forecast: ForecastConfig
    scenario: ScenarioConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    location: LocationConfig = Field(default_factory=LocationConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)


def load_project_config(path: Path) -> ProjectConfig:
    return ProjectConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
