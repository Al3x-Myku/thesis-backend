from sqlmodel import Session
from typing import Optional
from app.models.scene import Scene
from typing import List
from sqlmodel import Session, select
from app.models.scene import Scene

def create_scene_record(
    db: Session, owner_id: int, input_path: str
) -> Scene:
    """
    Insert a new Scene row (with placeholder input_path if needed)
    and return the Scene (populated with its new .id).
    """
    scene = Scene(owner_id=owner_id, input_path=input_path)
    db.add(scene)
    db.commit()
    db.refresh(scene)
    return scene

def get_scene(db: Session, scene_id: int) -> Optional[Scene]:
    return db.get(Scene, scene_id)

def get_scenes_by_owner(db: Session, owner_id: int) -> List[Scene]:
    stmt = select(Scene).where(Scene.owner_id == owner_id)
    return db.exec(stmt).all()