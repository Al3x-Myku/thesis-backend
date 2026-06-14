from typing import Optional
from datetime import datetime
from enum import Enum
from sqlmodel import SQLModel, Field

class DesignStatus(str, Enum):
    PENDING    = "PENDING"
    ANALYZING  = "ANALYZING"
    READY      = "READY"
    GENERATING = "GENERATING"
    COMMITTING = "COMMITTING"
    COMMITTED  = "COMMITTED"
    FAILED     = "FAILED"

class DesignSession(SQLModel, table=True):
    """One mood-board-driven design run rooted at a Scene (the room photo)."""
    __tablename__ = "design_sessions"

    id: Optional[int] = Field(default=None, primary_key=True)
    scene_id: int = Field(foreign_key="scenes.id", index=True, nullable=False)
    status: DesignStatus = Field(default=DesignStatus.PENDING, nullable=False, index=True)
    progress: float = Field(default=0.0, nullable=False)
    # Serialized DesignSpec (palette + furniture) produced by mood board analysis.
    design_spec_json: Optional[str] = None
    committed_variant_id: Optional[int] = None
    committed_glb_path: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Variant(SQLModel, table=True):
    """A single 2D iteration of a room within a DesignSession."""
    __tablename__ = "variants"

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="design_sessions.id", index=True, nullable=False)
    variant_index: int = Field(default=0)
    seed: int = Field(default=0)
    image_path: str
    # Which fields were frozen when this variant was produced: {"palette": bool, "slots": {id: bool}}
    locked_fields_json: str = Field(default="{}")
    # The DesignSpec snapshot actually used for THIS variant (for reproducibility / lineage).
    spec_snapshot_json: Optional[str] = None
    parent_variant_id: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ElementFeedback(SQLModel, table=True):
    """Per-element like/dislike. liked=True locks the element; False marks it for resampling."""
    __tablename__ = "element_feedback"

    id: Optional[int] = Field(default=None, primary_key=True)
    variant_id: int = Field(foreign_key="variants.id", index=True, nullable=False)
    element_type: str = Field(index=True)   # "palette" | "slot"
    element_key: str = Field(default="")     # "" for palette, slot id for slots
    liked: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
