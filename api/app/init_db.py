import time
from sqlalchemy.exc import OperationalError
from .db import Base, engine
from . import models  # noqa: F401

def init(retries: int = 10, delay: int = 2):
    for attempt in range(retries):
        try:
            Base.metadata.create_all(bind=engine)
            return
        except OperationalError as e:
            if attempt == retries - 1:
                raise
            print(f"DB not ready, retrying ({attempt + 1}/{retries})...")
            time.sleep(delay)
