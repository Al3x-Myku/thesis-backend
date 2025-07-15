import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

celery_app = Celery(
    "recon", 
    broker=f"sqla+{DATABASE_URL}",   
    backend=f"db+{DATABASE_URL}",    
    include=["app.tasks"],           
)

celery_app.conf.database_create_tables_at_setup = True