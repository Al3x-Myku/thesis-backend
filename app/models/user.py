from typing import Optional, List
from datetime import datetime
from sqlmodel import SQLModel, Field, Relationship
from pydantic import EmailStr

class UserBase(SQLModel):
    username: str = Field(index=True, nullable=False, unique=True)
    email: EmailStr   = Field(index=True, nullable=False, unique=True)
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str = Field(min_length=8, description="Plain-text password")

class UserRead(UserBase):
    id: int
    created_at: datetime

class User(UserBase, table=True):
    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    hashed_password: str = Field(nullable=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    scenes: List["Scene"] = Relationship(back_populates="owner")
