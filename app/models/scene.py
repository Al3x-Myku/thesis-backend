from typing import Optional
from datetime import datetime
from enum import Enum
from sqlmodel import SQLModel, Field, Relationship

class SceneStatus(str, Enum):
    PENDING     = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED   = "COMPLETED"
    FAILED      = "FAILED"

class SceneBase(SQLModel):
    owner_id: int = Field(foreign_key="users.id", nullable=False)
    input_path: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Scene(SceneBase, table=True):
    __tablename__ = "scenes"

    id: Optional[int] = Field(default=None, primary_key=True)
    status: SceneStatus = Field(default=SceneStatus.PENDING, nullable=False, index=True)
    progress: float = Field(default=0.0, nullable=False)
    result_path: Optional[str] = None

    owner: Optional["User"] = Relationship(back_populates="scenes")

class SceneRead(SceneBase):
    id: int
    status: SceneStatus
    progress: float
    result_path: Optional[str]
