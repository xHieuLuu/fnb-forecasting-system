from collections.abc import Sequence


class ForecastingDomainError(Exception):
    """Base error for failures surfaced by the forecasting domain."""


class DataValidationError(ForecastingDomainError):
    def __init__(
        self, table: str, columns: Sequence[str], rows: Sequence[int], reason: str
    ) -> None:
        self.table = table
        self.columns = list(columns)
        self.rows = list(rows)
        self.reason = reason
        super().__init__(f"{table}: {reason}; columns={self.columns}; rows={self.rows}")


class WeatherUnavailableError(ForecastingDomainError):
    """Raised when a weather provider cannot supply requested observations."""


class ArtifactCompatibilityError(ForecastingDomainError):
    """Raised when an artifact cannot be used by the current project."""


class ModelInferenceError(ForecastingDomainError):
    """Raised when a forecasting model cannot generate predictions."""
