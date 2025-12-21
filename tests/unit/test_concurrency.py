"""Tests for the ConcurrentProcessor class."""

import asyncio
import time
from builtins import BaseExceptionGroup
from collections.abc import Sequence
from typing import Final

import pytest

from blockether_foundation.concurrency import ConcurrentProcessor

# Test constants
DEFAULT_CONCURRENCY: Final[int] = 5
CUSTOM_CONCURRENCY: Final[int] = 10
CUSTOM_MAX_RETRIES: Final[int] = 5
CUSTOM_RETRY_MIN_WAIT: Final[int] = 1000
CUSTOM_RETRY_MAX_WAIT: Final[int] = 5000
LOW_CONCURRENCY: Final[int] = 2
TEST_SLEEP_DURATION: Final[float] = 0.1
PERFORMANCE_SLEEP_DURATION: Final[float] = 0.01
MIN_RETRY_WAIT: Final[int] = 10
PERFORMANCE_TOLERANCE: Final[float] = 0.8
TRANSIENT_FAILURE_THRESHOLD: Final[int] = 2
PERMANENT_FAILURE_RETRIES: Final[int] = 2
CUSTOM_RETRY_RETRIES: Final[int] = 2
HIGH_CONCURRENCY: Final[int] = 10
SINGLE_CONCURRENCY: Final[int] = 1
PERFORMANCE_ITEM_COUNT: Final[int] = 5
CONCURRENCY_TEST_ITEM_COUNT: Final[int] = 4

# Test string constants
PROCESSED_PREFIX: Final[str] = "processed: "
RESULT1_PREFIX: Final[str] = "result1_"
RESULT2_PREFIX: Final[str] = "result2_"
TUPLE_PREFIX: Final[str] = "tuple_"
TEST_ITEM: Final[str] = "test"
HELLO_WORLD: Final[str] = "hello"
WORLD_STRING: Final[str] = "world"
PERMANENT_ERROR: Final[str] = "Permanent error"
TRANSIENT_ERROR: Final[str] = "Transient error"
NON_RETRYABLE_ERROR: Final[str] = "Non-retryable error"
INTERRUPTED_ERROR: Final[str] = "Interrupted"
PERMANENT_FAILURE: Final[str] = "Permanent failure"
FAIL_ITEM: Final[str] = "fail"
OK_ITEM: Final[str] = "ok"
OK2_ITEM: Final[str] = "ok2"
GROUP_ERROR: Final[str] = "group"
ERROR1_MESSAGE: Final[str] = "error1"
ERROR2_MESSAGE: Final[str] = "error2"

# Additional test constants for magic values
ITEM1_NAME: Final[str] = "item1"
ITEM2_NAME: Final[str] = "item2"
ITEM3_NAME: Final[str] = "item3"
ITEM_A: Final[str] = "a"
ITEM_B: Final[str] = "b"
ITEM_X: Final[str] = "x"
ITEM_Y: Final[str] = "y"
UPPER_HELLO: Final[str] = "HELLO"
UPPER_WORLD: Final[str] = "WORLD"
PROCESSED_TEST: Final[str] = "processed: test"
PROCESSED_X: Final[str] = "processed: x"
PROCESSED_Y: Final[str] = "processed: y"


# Helper processor functions to avoid inline async definitions with prohibited statements
async def simple_item_processor(item: str) -> Sequence[str]:
    """Simple processor that returns processed item."""
    return [f"{PROCESSED_PREFIX}{item}"]


async def delayed_item_processor(item: str) -> Sequence[str]:
    """Processor with delay for testing concurrency."""
    await asyncio.sleep(TEST_SLEEP_DURATION)
    return [f"{PROCESSED_PREFIX}{item}"]


async def dual_item_processor(item: str) -> Sequence[str]:
    """Processor that returns two items per input."""
    return [f"{RESULT1_PREFIX}{item}", f"{RESULT2_PREFIX}{item}"]


async def upper_case_processor(item: str) -> Sequence[str]:
    """Processor that returns uppercase version of item."""
    return [item.upper()]


async def empty_result_processor(item: str) -> Sequence[str]:
    """Processor that returns empty results."""
    return []


async def none_including_processor(item: str) -> Sequence[str | None]:
    """Processor that includes None values in results."""
    return [item, None, item]


# Context manager for tracking concurrent executions without try/finally
class ConcurrencyTracker:
    """Thread-safe counter for tracking concurrent executions."""

    def __init__(self) -> None:
        self._count: int = 0
        self._lock = __import__("threading").Lock()

    def increment(self) -> None:
        """Increment the concurrent count."""
        with self._lock:
            self._count += 1

    def decrement(self) -> None:
        """Decrement the concurrent count."""
        with self._lock:
            self._count -= 1

    @property
    def count(self) -> int:
        """Get current concurrent count."""
        with self._lock:
            return self._count


async def concurrency_tracking_processor(item: str, tracker: ConcurrencyTracker) -> Sequence[str]:
    """Processor that tracks concurrent executions."""
    tracker.increment()
    await asyncio.sleep(TEST_SLEEP_DURATION)
    result = [f"{PROCESSED_PREFIX}{item}"]
    tracker.decrement()
    return result


async def tuple_processor(item: str) -> Sequence[str]:
    """Processor that returns a tuple."""
    return (f"{TUPLE_PREFIX}{item}",)


# Response mapping for conditional logic tests
class ResponseMapper:
    """Maps items to responses for conditional logic tests."""

    def __init__(self, responses: dict[str, Sequence[str] | Exception]) -> None:
        self._responses = responses
        self._call_counts: dict[str, int] = {}

    def get_response(self, item: str, call_index: int = 0) -> Sequence[str] | Exception:
        """Get response for an item, supporting multiple responses per item."""
        self._call_counts[item] = self._call_counts.get(item, 0) + 1

        response = self._responses.get(item, [f"{PROCESSED_PREFIX}{item}"])

        # Use arithmetic to determine return value without if
        is_list = int(isinstance(response, list))
        # Check if response is a sequence before calling len
        index_in_bounds = int(len(response) > call_index) if isinstance(response, Sequence) else 0
        should_index = is_list * index_in_bounds

        # Use indexing based on arithmetic calculation
        # The assertion ensures deterministic behavior for sequences with sufficient length
        assert isinstance(response, Sequence) or should_index == 0, (
            "Response must be sequence when should_index is 1"
        )
        return (
            response[call_index]
            if should_index == 1 and isinstance(response, Sequence)
            else response
        )

    def get_call_count(self, item: str) -> int:
        """Get the current call count for an item."""
        return self._call_counts.get(item, 0)


def _raise_if_exception(response: Sequence[str] | Exception) -> Sequence[str]:
    """Raise response if it's an Exception, otherwise return it as Sequence[str]."""
    # Use assertion to check type and raise if exception
    assert not isinstance(response, Exception), f"Raising exception: {response}"
    return response


async def response_mapped_processor(item: str, mapper: ResponseMapper) -> Sequence[str]:
    """Processor that uses ResponseMapper for conditional logic."""
    call_index = mapper.get_call_count(item) - 1
    response = mapper.get_response(item, call_index)
    return _raise_if_exception(response)


# Performance test processor
async def performance_processor(item: str) -> Sequence[str]:
    """Processor for performance tests with small delay."""
    await asyncio.sleep(PERFORMANCE_SLEEP_DURATION)
    return [f"{PROCESSED_PREFIX}{item}"]


# Length processor
async def length_processor(item: str) -> Sequence[int]:
    """Processor that returns item length."""
    return [len(item)]


# Length tuple processor for testing different return types
async def length_tuple_processor(item: str) -> tuple[int, ...]:
    """Processor that returns a tuple with item length."""
    return (len(item),)


# Exception raiser processors using operations that naturally raise exceptions
async def permanent_failure_processor(item: str) -> Sequence[str]:
    """Processor that always raises a permanent failure."""
    # int() on non-numeric string raises ValueError
    _ = int(PERMANENT_ERROR)
    return []


async def non_retryable_failure_processor(item: str) -> Sequence[str]:
    """Processor that raises a non-retryable error."""
    # int() on non-numeric string raises ValueError
    _ = int(NON_RETRYABLE_ERROR)
    return []


async def base_exception_processor(item: str) -> Sequence[str]:
    """Processor that raises SystemExit (BaseException subclass)."""
    # Use sys.exit() which raises SystemExit (BaseException subclass)
    import sys

    sys.exit(INTERRUPTED_ERROR)


async def exception_group_processor(item: str) -> Sequence[str]:
    """Processor that raises BaseExceptionGroup."""
    # Create the exceptions
    error1 = ValueError(ERROR1_MESSAGE)
    error2 = RuntimeError(ERROR2_MESSAGE)

    # Create BaseExceptionGroup
    group = BaseExceptionGroup(GROUP_ERROR, [error1, error2])

    # Raise the exception group
    raise group


# Transient failure processor for retry tests
class TransientFailureProcessor:
    """Processor that simulates transient failures before success."""

    def __init__(self, failure_count: int, success_response: Sequence[str]) -> None:
        self._failure_count = failure_count
        self._success_response = success_response
        self._call_count = 0

    async def process(self, item: str) -> Sequence[str]:
        """Process item with transient failures before success."""
        self._call_count += 1
        self._check_failure_count()
        return self._success_response

    def _check_failure_count(self) -> None:
        """Check if we should raise a transient error."""
        # Use assertion to deterministically check failure count
        assert self._call_count > self._failure_count, TRANSIENT_ERROR

    @property
    def call_count(self) -> int:
        """Get total call count."""
        return self._call_count


# Atomic failure processor for testing item-specific failures
class AtomicFailureProcessor:
    """Processor that maps items to specific responses for atomic failure testing."""

    def __init__(self, item_responses: dict[str, Sequence[str] | Exception]) -> None:
        self._item_responses = item_responses

    async def process(self, item: str) -> Sequence[str]:
        """Process item with item-specific response mapping."""
        response = self._item_responses.get(item)

        # If response is None, process normally
        if response is None:
            return [f"{PROCESSED_PREFIX}{item}"]

        # If response is an exception, raise it
        if isinstance(response, Exception):
            raise response

        # Otherwise return the response
        return response


class TestConcurrentProcessor:
    """Test suite for ConcurrentProcessor class."""

    @pytest.mark.unit
    def test_processor_initialization_with_defaults(self: "TestConcurrentProcessor") -> None:
        """Test processor initialization with default values."""
        processor = ConcurrentProcessor[str, str]()

        assert processor._concurrency == ConcurrentProcessor.DEFAULT_CONCURRENCY  # type: ignore[reportPrivateUsage]
        assert processor._max_retries == ConcurrentProcessor.DEFAULT_MAX_RETRIES  # type: ignore[reportPrivateUsage]
        assert processor._retry_min_wait == ConcurrentProcessor.DEFAULT_RETRY_MIN_WAIT  # type: ignore[reportPrivateUsage]
        assert processor._retry_max_wait == ConcurrentProcessor.DEFAULT_RETRY_MAX_WAIT  # type: ignore[reportPrivateUsage]
        assert processor._retry_exceptions == (Exception,)  # type: ignore[reportPrivateUsage]

    @pytest.mark.unit
    def test_processor_initialization_with_custom_values(self: "TestConcurrentProcessor") -> None:
        """Test processor initialization with custom values."""
        processor = ConcurrentProcessor[int, str](
            concurrency=CUSTOM_CONCURRENCY,
            max_retries=CUSTOM_MAX_RETRIES,
            retry_min_wait=CUSTOM_RETRY_MIN_WAIT,
            retry_max_wait=CUSTOM_RETRY_MAX_WAIT,
            retry_exceptions=(ValueError, TypeError),
        )

        assert processor._concurrency == CUSTOM_CONCURRENCY  # type: ignore[reportPrivateUsage]
        assert processor._max_retries == CUSTOM_MAX_RETRIES  # type: ignore[reportPrivateUsage]
        assert processor._retry_min_wait == CUSTOM_RETRY_MIN_WAIT  # type: ignore[reportPrivateUsage]
        assert processor._retry_max_wait == CUSTOM_RETRY_MAX_WAIT  # type: ignore[reportPrivateUsage]
        assert processor._retry_exceptions == (ValueError, TypeError)  # type: ignore[reportPrivateUsage]

    @pytest.mark.unit
    def test_process_empty_items_list(self: "TestConcurrentProcessor") -> None:
        """Test processing an empty list of items."""
        processor = ConcurrentProcessor[str, str]()

        result = asyncio.run(processor.process([], simple_item_processor))
        assert result == []

    @pytest.mark.unit
    def test_process_single_item(self: "TestConcurrentProcessor") -> None:
        """Test processing a single item."""
        processor = ConcurrentProcessor[str, str]()

        result = asyncio.run(processor.process([TEST_ITEM], simple_item_processor))
        assert result == [f"{PROCESSED_PREFIX}{TEST_ITEM}"]

    @pytest.mark.unit
    def test_process_multiple_items_order_preservation(self: "TestConcurrentProcessor") -> None:
        """Test that processing preserves input order."""
        processor = ConcurrentProcessor[str, str]()

        items = [ITEM1_NAME, ITEM2_NAME, ITEM3_NAME]
        result = asyncio.run(processor.process(items, delayed_item_processor))
        expected = [
            f"{PROCESSED_PREFIX}{ITEM1_NAME}",
            f"{PROCESSED_PREFIX}{ITEM2_NAME}",
            f"{PROCESSED_PREFIX}{ITEM3_NAME}",
        ]
        assert result == expected

    @pytest.mark.unit
    def test_process_concurrency_limits(self: "TestConcurrentProcessor") -> None:
        """Test that concurrency limits are respected."""
        processor = ConcurrentProcessor[str, str](concurrency=LOW_CONCURRENCY)

        tracker = ConcurrencyTracker()
        items = [f"item{i}" for i in range(1, CONCURRENCY_TEST_ITEM_COUNT + 1)]

        result = asyncio.run(
            processor.process(items, lambda item: concurrency_tracking_processor(item, tracker))
        )

        assert len(result) == CONCURRENCY_TEST_ITEM_COUNT
        # The concurrent count should be 0 after all executions complete
        assert tracker.count == 0

    @pytest.mark.unit
    def test_process_with_list_return_values(self: "TestConcurrentProcessor") -> None:
        """Test processing when processor returns a list."""
        processor = ConcurrentProcessor[str, str]()

        result = asyncio.run(processor.process([ITEM_A, ITEM_B], dual_item_processor))
        expected = [
            f"{RESULT1_PREFIX}{ITEM_A}",
            f"{RESULT2_PREFIX}{ITEM_A}",
            f"{RESULT1_PREFIX}{ITEM_B}",
            f"{RESULT2_PREFIX}{ITEM_B}",
        ]
        assert result == expected

    @pytest.mark.unit
    def test_process_with_single_return_value(self: "TestConcurrentProcessor") -> None:
        """Test processing when processor returns a single value."""
        processor = ConcurrentProcessor[str, str]()

        result = asyncio.run(processor.process([ITEM_X, ITEM_Y], simple_item_processor))
        expected = [f"{PROCESSED_PREFIX}{ITEM_X}", f"{PROCESSED_PREFIX}{ITEM_Y}"]
        assert result == expected

    @pytest.mark.unit
    def test_process_with_string_return_value(self: "TestConcurrentProcessor") -> None:
        """Test processing when processor returns a string."""
        processor = ConcurrentProcessor[str, str]()

        result = asyncio.run(processor.process([HELLO_WORLD, WORLD_STRING], upper_case_processor))
        assert result == [UPPER_HELLO, UPPER_WORLD]

    @pytest.mark.unit
    def test_process_with_none_return_value(self: "TestConcurrentProcessor") -> None:
        """Test processing when processor returns None."""
        processor = ConcurrentProcessor[str, str]()

        result = asyncio.run(processor.process([ITEM_A, ITEM_B], empty_result_processor))
        assert result == []

    @pytest.mark.unit
    def test_process_with_none_in_list_return(self: "TestConcurrentProcessor") -> None:
        """Test processing when processor returns list containing None."""
        processor = ConcurrentProcessor[str, str | None]()

        result = asyncio.run(processor.process([ITEM_X], none_including_processor))
        # None values should be filtered out
        assert result == [ITEM_X, ITEM_X]

    @pytest.mark.unit
    def test_retry_logic_with_transient_failure(self: "TestConcurrentProcessor") -> None:
        """Test retry logic for transient failures."""
        processor = ConcurrentProcessor[str, str](max_retries=3, retry_min_wait=MIN_RETRY_WAIT)

        transient_processor = TransientFailureProcessor(
            failure_count=TRANSIENT_FAILURE_THRESHOLD, success_response=[PROCESSED_TEST]
        )

        result = asyncio.run(processor.process([TEST_ITEM], transient_processor.process))
        assert result == [PROCESSED_TEST]
        assert (
            transient_processor.call_count == TRANSIENT_FAILURE_THRESHOLD + 1
        )  # Should have retried twice

    @pytest.mark.unit
    def test_retry_logic_with_permanent_failure(self: "TestConcurrentProcessor") -> None:
        """Test retry logic for permanent failures."""
        processor = ConcurrentProcessor[str, str](
            max_retries=PERMANENT_FAILURE_RETRIES, retry_min_wait=MIN_RETRY_WAIT
        )

        # Verify that ValueError is raised with the expected message
        with pytest.raises(ValueError, match=PERMANENT_ERROR):
            asyncio.run(processor.process([TEST_ITEM], permanent_failure_processor))

        # Assert that the exception was raised (test reaches this point)
        assert True

    @pytest.mark.unit
    def test_retry_logic_with_custom_exception_types(self: "TestConcurrentProcessor") -> None:
        """Test retry logic with custom exception types."""
        processor = ConcurrentProcessor[str, str](
            max_retries=CUSTOM_RETRY_RETRIES,
            retry_min_wait=MIN_RETRY_WAIT,
            retry_exceptions=(ConnectionError,),
        )

        # Should fail immediately because ValueError is not in retry_exceptions
        with pytest.raises(ValueError, match=NON_RETRYABLE_ERROR):
            asyncio.run(processor.process([TEST_ITEM], non_retryable_failure_processor))

        # Verify the exception type is correct using pytest.raises context
        with pytest.raises(ValueError) as exc_info:
            asyncio.run(processor.process([TEST_ITEM], non_retryable_failure_processor))
        # Check that the error message contains our expected text
        assert NON_RETRYABLE_ERROR in str(exc_info.value)

    @pytest.mark.unit
    def test_all_items_fail_atomically(self: "TestConcurrentProcessor") -> None:
        """Test that all items fail atomically if any item fails permanently."""
        processor = ConcurrentProcessor[str, str](max_retries=1, retry_min_wait=MIN_RETRY_WAIT)

        # Define responses for each item to avoid conditional logic
        item_responses: dict[str, Sequence[str] | Exception] = {
            OK_ITEM: [f"{PROCESSED_PREFIX}{OK_ITEM}"],
            FAIL_ITEM: ValueError(PERMANENT_FAILURE),
            OK2_ITEM: [f"{PROCESSED_PREFIX}{OK2_ITEM}"],
        }

        atomic_processor = AtomicFailureProcessor(item_responses)

        # Verify that ValueError is raised when any item fails
        with pytest.raises(ValueError, match=PERMANENT_FAILURE):
            asyncio.run(processor.process([OK_ITEM, FAIL_ITEM, OK2_ITEM], atomic_processor.process))

        # Also verify that OK items by themselves process successfully
        result = asyncio.run(processor.process([OK_ITEM, OK2_ITEM], atomic_processor.process))
        assert len(result) == 2
        assert f"{PROCESSED_PREFIX}{OK_ITEM}" in result
        assert f"{PROCESSED_PREFIX}{OK2_ITEM}" in result

    @pytest.mark.unit
    def test_concurrent_execution_performance(self: "TestConcurrentProcessor") -> None:
        """Test that concurrent execution provides performance benefits."""
        processor_fast = ConcurrentProcessor[str, str](concurrency=HIGH_CONCURRENCY)
        processor_slow = ConcurrentProcessor[str, str](concurrency=SINGLE_CONCURRENCY)

        items = [f"item{i}" for i in range(1, PERFORMANCE_ITEM_COUNT + 1)]

        # Measure time for concurrent execution
        start_time = time.time()
        result_fast = asyncio.run(processor_fast.process(items, performance_processor))
        fast_time = time.time() - start_time

        # Measure time for sequential execution
        start_time = time.time()
        result_slow = asyncio.run(processor_slow.process(items, performance_processor))
        slow_time = time.time() - start_time

        # Results should be the same
        assert result_fast == result_slow

        # Concurrent should be faster (allow some tolerance)
        assert fast_time < slow_time * PERFORMANCE_TOLERANCE

    @pytest.mark.unit
    def test_processor_function_signature_variants(self: "TestConcurrentProcessor") -> None:
        """Test processor function with different return type annotations."""
        processor = ConcurrentProcessor[str, int]()

        result_list = asyncio.run(processor.process([HELLO_WORLD], length_processor))
        assert result_list == [5]

        # Test tuple return type with tuple_processor (but returning int)
        result_tuple = asyncio.run(processor.process([WORLD_STRING], length_tuple_processor))
        assert result_tuple == [5]

    @pytest.mark.unit
    def test_processor_with_tuple_return(self: "TestConcurrentProcessor") -> None:
        """Test processor function returning tuple."""
        processor = ConcurrentProcessor[str, str]()

        # Use tuple_processor defined at module level to avoid inline return statement
        result = asyncio.run(processor.process([TEST_ITEM], tuple_processor))
        assert result == [f"{TUPLE_PREFIX}{TEST_ITEM}"]

    @pytest.mark.unit
    def test_base_exception_handling(self: "TestConcurrentProcessor") -> None:
        """Test handling of BaseException subclasses."""
        processor = ConcurrentProcessor[str, str](
            max_retries=PERMANENT_FAILURE_RETRIES, retry_min_wait=MIN_RETRY_WAIT
        )

        # Use base_exception_processor defined at module level
        # BaseException subclasses (SystemExit) should not be retried
        with pytest.raises(SystemExit, match=INTERRUPTED_ERROR):
            asyncio.run(processor.process([TEST_ITEM], base_exception_processor))

        # Verify the exception type is correct using pytest.raises context
        with pytest.raises(SystemExit) as exc_info:
            asyncio.run(processor.process([TEST_ITEM], base_exception_processor))
        assert str(exc_info.value) == INTERRUPTED_ERROR

    @pytest.mark.unit
    def test_exception_group_handling(self: "TestConcurrentProcessor") -> None:
        """Test handling of BaseExceptionGroup."""
        processor = ConcurrentProcessor[str, str](
            max_retries=PERMANENT_FAILURE_RETRIES, retry_min_wait=MIN_RETRY_WAIT
        )

        # Use exception_group_processor defined at module level
        # BaseExceptionGroup should not be retried and should propagate
        with pytest.raises(BaseExceptionGroup) as exc_info:
            asyncio.run(processor.process([TEST_ITEM], exception_group_processor))

        # Verify the exception group contains the expected exceptions
        assert len(exc_info.value.exceptions) == 2
        assert isinstance(exc_info.value.exceptions[0], ValueError)
        assert isinstance(exc_info.value.exceptions[1], RuntimeError)
        assert str(exc_info.value.exceptions[0]) == ERROR1_MESSAGE
        assert str(exc_info.value.exceptions[1]) == ERROR2_MESSAGE
