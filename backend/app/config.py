from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_path: str = "models/best_phone_bag_hard_negative_yolo11s.pt"
    confidence_threshold: float = 0.30
    dangerous_confidence_threshold: float = 0.55
    iou_threshold: float = 0.50
    alert_gap_seconds: float = 1.5
    frame_skip: int = 1
    max_upload_bytes: int = 500 * 1024 * 1024
    upload_dir: Path = Path("data/uploads")
    output_dir: Path = Path("data/outputs")
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000"
    gmail_address: str | None = None
    gmail_app_password: str | None = None
    alert_recipient: str | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def resolved_model_path(self) -> Path:
        model_path = Path(self.model_path)
        if model_path.is_absolute():
            return model_path
        return Path(__file__).resolve().parents[1] / model_path


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.output_dir.mkdir(parents=True, exist_ok=True)
