import os, pathlib, pytest
os.environ["ALTO_MINI_DB"] = str(pathlib.Path(__file__).resolve().parent / "_test.db")
from app import data
@pytest.fixture(autouse=True, scope="session")
def seeded():
    data.seed(days=5); yield
    pathlib.Path(os.environ["ALTO_MINI_DB"]).unlink(missing_ok=True)
@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)
