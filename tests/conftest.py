import pytest


@pytest.fixture(scope="session")
def anyio_backend():
    print(111111111)
    print(111111111)
    print(111111111)
    return "asyncio"
