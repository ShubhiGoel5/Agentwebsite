"""Worker Mock Environments Package."""

from worker.env.mock_erp import MockERPEnvironment
from worker.env.mock_mail import MockMailEnvironment

__all__ = ["MockMailEnvironment", "MockERPEnvironment"]
