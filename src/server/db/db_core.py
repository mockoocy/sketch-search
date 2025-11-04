from sqlmodel import SQLModel, create_engine, Session

_engine = create_engine("sqlite:///./app.db", connect_args={"check_same_thread": False})

def init_db() -> None:
    SQLModel.metadata.create_all(_engine)

def get_session():
    with Session(_engine) as s:
        yield s