from fastapi import APIRouter, Depends, status
from sqlmodel import Session
from app.services.user_service import register_user, get_user
from app.models.user import UserCreate, UserRead
from app.database import get_db  # see below for this helper
from app.core.security import create_access_token
from app.schemas.token import Token as SignupToken


router = APIRouter(prefix="/users", tags=["users"])

@router.post(
    "/",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
)
def create_user(
    user_in: UserCreate,
    db: Session = Depends(get_db),
):
    """
    Register a new user.
    - Validates username/email uniqueness
    - Hashes password
    - Returns the created UserRead schema
    """
    return register_user(db, user_in)

@router.get(
    "/{user_id}",
    response_model=UserRead,
    status_code=status.HTTP_200_OK,
)
def read_user(
    user_id: int,
    db: Session = Depends(get_db),
):
    """
    Fetch a user by ID.
    Raises 404 if not found.
    """
    return get_user(db, user_id)

@router.post(
    "/signup",
    response_model=SignupToken,
    status_code=status.HTTP_201_CREATED,
)
def signup(
    user_in: UserCreate,
    db: Session = Depends(get_db),
):
    user_read = register_user(db, user_in)
    token = create_access_token(data={"sub": str(user_read.id)})
    return {"access_token": token, "token_type": "bearer"}