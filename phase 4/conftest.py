import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (real API calls, requires services)")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
