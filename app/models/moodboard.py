from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field

class MoodBoardImage(SQLModel, table=True):
    __tablename__ = "moodboard_images"

    id: Optional[int] = Field(default=None, primary_key=True)
    scene_id: int = Field(foreign_key="scenes.id", index=True, nullable=False)
    file_path: str
    upload_order: int = Field(default=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
