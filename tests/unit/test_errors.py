
from blockether_foundation.errors import FoundationBaseError


class CustomTestError(FoundationBaseError):
    pass


def test_error_having_auto_solidity_like_message():
    error = CustomTestError("Test error occurred")
    assert str(error) == "test_errors.CustomTestError: Test error occurred"


def test_error_with_details():
    """Test error string representation with details (covers lines 33-34)."""
    from pydantic import BaseModel

    class ErrorDetails(BaseModel):
        code: int
        field: str

    error = CustomTestError("Validation failed", details=ErrorDetails(code=400, field="username"))

    error_str = str(error)
    assert "test_errors.CustomTestError: Validation failed" in error_str
    assert "details:" in error_str
    assert "code" in error_str
    assert "400" in error_str
    assert "field" in error_str
    assert "username" in error_str


def test_consensus_field_init_error():
    """Test ConsensusFieldInitError functionality (covers line 41)."""
    from blockether_foundation.errors import ConsensusFieldInitError

    error = ConsensusFieldInitError("Field initialization failed")
    error_str = str(error)
    assert (
        "blockether_foundation.errors.ConsensusFieldInitError: Field initialization failed"
        in error_str
    )
    assert hasattr(error, "timestamp")
    assert hasattr(error, "message")
    assert error.message == "Field initialization failed"
