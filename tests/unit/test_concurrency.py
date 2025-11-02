"""Tests for the ConcurrentProcessor class."""

import asyncio
import time
from collections.abc import Sequence

import pytest

from blockether_foundation.concurrency import ConcurrentProcessor


class TestConcurrentProcessor:
    """Test suite for ConcurrentProcessor class."""

    def test_processor_initialization_with_defaults(self):
        """Test processor initialization with default values."""
        processor = ConcurrentProcessor[str, str]()

        assert processor._concurrency == processor.DEFAULT_CONCURRENCY
        assert processor._max_retries == processor.DEFAULT_MAX_RETRIES
        assert processor._retry_min_wait == processor.DEFAULT_RETRY_MIN_WAIT
        assert processor._retry_max_wait == processor.DEFAULT_RETRY_MAX_WAIT
        assert processor._retry_exceptions == (Exception,)

    def test_processor_initialization_with_custom_values(self):
        """Test processor initialization with custom values."""
        processor = ConcurrentProcessor[int, str](
            concurrency=10,
            max_retries=5,
            retry_min_wait=1000,
            retry_max_wait=20000,
            retry_exceptions=(ValueError, TypeError),
        )

        assert processor._concurrency == 10
        assert processor._max_retries == 5
        assert processor._retry_min_wait == 1000
        assert processor._retry_max_wait == 20000
        assert processor._retry_exceptions == (ValueError, TypeError)

    async def test_process_empty_items_list(self):
        """Test processing an empty list of items."""
        processor = ConcurrentProcessor[str, str]()

        async def mock_processor(item: str) -> Sequence[str]:
            return [f"processed_{item}"]

        result = await processor.process([], mock_processor)
        assert result == []

    async def test_process_single_item(self):
        """Test processing a single item."""
        processor = ConcurrentProcessor[str, str]()

        async def mock_processor(item: str) -> Sequence[str]:
            return [f"processed_{item}"]

        result = await processor.process(["test"], mock_processor)
        assert result == ["processed_test"]

    async def test_process_multiple_items_order_preservation(self):
        """Test that results maintain the same order as inputs."""
        processor = ConcurrentProcessor[int, str](concurrency=3)

        async def mock_processor(item: int) -> Sequence[str]:
            # Add delay to test concurrent execution
            await asyncio.sleep(0.1)
            return [f"item_{item}"]

        items = [3, 1, 4, 1, 5]
        result = await processor.process(items, mock_processor)
        expected = ["item_3", "item_1", "item_4", "item_1", "item_5"]
        assert result == expected

    async def test_process_concurrency_limits(self):
        """Test that concurrency limits are respected."""
        processor = ConcurrentProcessor[str, str](concurrency=2)

        # Track concurrent executions
        active_count = 0
        max_active = 0
        active_lock = asyncio.Lock()

        async def mock_processor(item: str) -> Sequence[str]:
            nonlocal active_count, max_active

            async with active_lock:
                active_count += 1
                max_active = max(max_active, active_count)

            # Simulate work
            await asyncio.sleep(0.1)

            async with active_lock:
                active_count -= 1

            return [f"processed_{item}"]

        items = ["a", "b", "c", "d", "e"]
        await processor.process(items, mock_processor)

        # Verify concurrency limit was not exceeded
        assert max_active <= 2

    async def test_process_with_list_return_values(self):
        """Test processor that returns lists of values."""
        processor = ConcurrentProcessor[int, str]()

        async def mock_processor(item: int) -> Sequence[str]:
            return [f"value_{item}", f"double_{item * 2}"]

        result = await processor.process([1, 2], mock_processor)
        expected = ["value_1", "double_2", "value_2", "double_4"]
        assert result == expected

    async def test_process_with_single_return_value(self):
        """Test processor that returns single values (wrapped in list)."""
        processor = ConcurrentProcessor[str, int]()

        async def mock_processor(item: str) -> int:
            return len(item)

        result = await processor.process(["hello", "world"], mock_processor)
        assert result == [5, 5]

    async def test_process_with_string_return_value(self):
        """Test processor that returns string values."""
        processor = ConcurrentProcessor[int, str]()

        async def mock_processor(item: int) -> str:
            return f"number_{item}"

        result = await processor.process([1, 2], mock_processor)
        assert result == ["number_1", "number_2"]

    async def test_process_with_none_return_value(self):
        """Test processor that returns None (filtered out)."""
        processor = ConcurrentProcessor[int, str]()

        async def mock_processor(item: int) -> str | None:
            if item % 2 == 0:
                return f"even_{item}"
            return None

        result = await processor.process([1, 2, 3, 4], mock_processor)
        assert result == ["even_2", "even_4"]

    async def test_process_with_none_in_list_return(self):
        """Test processor that returns list containing None values (filtered out)."""
        processor = ConcurrentProcessor[int, str]()

        async def mock_processor(item: int) -> Sequence[str | None]:
            if item == 2:
                return ["valid_2", None]
            return [f"valid_{item}"]

        result = await processor.process([1, 2, 3], mock_processor)
        assert result == ["valid_1", "valid_2", "valid_3"]

    async def test_retry_logic_with_transient_failure(self):
        """Test retry logic with transient failures that eventually succeed."""
        processor = ConcurrentProcessor[str, str](
            max_retries=3,
            retry_min_wait=100,  # Shorter wait for tests
            retry_max_wait=200,
        )

        call_count = 0

        async def failing_processor(item: str) -> Sequence[str]:
            nonlocal call_count
            call_count += 1

            if call_count <= 2:  # Fail first 2 attempts
                raise ValueError(f"Temporary failure #{call_count}")

            return [f"success_{item}"]

        result = await processor.process(["test"], failing_processor)
        assert result == ["success_test"]
        assert call_count == 3  # Should have retried twice

    async def test_retry_logic_with_permanent_failure(self):
        """Test retry logic with permanent failures that never succeed."""
        processor = ConcurrentProcessor[str, str](
            max_retries=2,
            retry_min_wait=50,
            retry_max_wait=100,
        )

        async def always_failing_processor(item: str) -> Sequence[str]:
            raise ValueError("Permanent failure")

        with pytest.raises(ValueError, match="Permanent failure"):
            await processor.process(["test"], always_failing_processor)

    async def test_retry_logic_with_custom_exception_types(self):
        """Test retry logic with custom exception types."""
        processor = ConcurrentProcessor[str, str](
            max_retries=2,
            retry_min_wait=50,
            retry_max_wait=100,
            retry_exceptions=(ValueError,),  # Only retry ValueError
        )

        call_count = 0

        async def processor_with_multiple_exceptions(item: str) -> Sequence[str]:
            nonlocal call_count
            call_count += 1

            if item == "retry_me":
                if call_count == 1:  # Fail first attempt, succeed on 2nd
                    raise ValueError("This should be retried")
                return [f"success_{item}"]
            elif item == "dont_retry_me":
                raise TypeError("This should not be retried")
            return [f"success_{item}"]

        # Reset call count for this test
        call_count = 0
        # This should retry and eventually succeed
        result = await processor.process(["retry_me"], processor_with_multiple_exceptions)
        assert result == ["success_retry_me"]
        assert call_count == 2  # Should have retried once then succeeded

        # This should fail immediately without retries
        with pytest.raises(TypeError, match="This should not be retried"):
            await processor.process(["dont_retry_me"], processor_with_multiple_exceptions)

    async def test_all_items_fail_atomically(self):
        """Test that all items fail atomically - if one fails, all fail."""
        processor = ConcurrentProcessor[str, str](
            max_retries=1,
            retry_min_wait=50,
            retry_max_wait=100,
        )

        async def failing_processor(item: str) -> Sequence[str]:
            if item == "fail":
                raise ValueError("This item fails")
            return [f"success_{item}"]

        # All processing should fail even though only one item fails
        with pytest.raises(ValueError, match="This item fails"):
            await processor.process(["success1", "fail", "success2"], failing_processor)

    async def test_concurrent_execution_performance(self):
        """Test that concurrent execution is faster than sequential."""
        processor = ConcurrentProcessor[str, str](concurrency=3)

        async def slow_processor(item: str) -> Sequence[str]:
            await asyncio.sleep(0.1)  # 100ms delay per item
            return [f"processed_{item}"]

        items = ["a", "b", "c", "d", "e", "f"]

        start_time = time.time()
        await processor.process(items, slow_processor)
        concurrent_time = time.time() - start_time

        # With concurrency=3, should take approximately (6/3) * 0.1 = 0.2 seconds
        # Sequential would take 6 * 0.1 = 0.6 seconds
        # Allow some margin for test execution overhead
        assert concurrent_time < 0.5  # Should be much less than sequential

    async def test_processor_function_signature_variants(self):
        """Test processor functions with different return type signatures."""
        processor = ConcurrentProcessor[str, str]()

        # Test returning Sequence[TOutput | None]
        async def sequence_processor(item: str) -> Sequence[str | None]:
            return [f"seq_{item}"]

        # Test returning TOutput | None
        async def single_processor(item: str) -> str | None:
            return f"single_{item}"

        # Test returning TOutput
        async def value_processor(item: str) -> str:
            return f"value_{item}"

        result1 = await processor.process(["test1"], sequence_processor)
        result2 = await processor.process(["test2"], single_processor)
        result3 = await processor.process(["test3"], value_processor)

        assert result1 == ["seq_test1"]
        assert result2 == ["single_test2"]
        assert result3 == ["value_test3"]

    async def test_processor_with_tuple_return(self):
        """Test processor that returns tuple (sequence type)."""
        processor = ConcurrentProcessor[int, str]()

        async def tuple_processor(item: int) -> tuple[str, ...]:
            return (f"tuple_{item}", f"double_{item * 2}")

        result = await processor.process([1, 2], tuple_processor)
        assert result == ["tuple_1", "double_2", "tuple_2", "double_4"]

    async def test_base_exception_handling(self):
        """Test handling of BaseException subclasses (not Exception)."""
        processor = ConcurrentProcessor[str, str]()

        class CustomBaseException(BaseException):
            """Custom BaseException subclass for testing."""

            pass

        async def processor_with_base_exception(item: str) -> Sequence[str]:
            # Raise a BaseException subclass that's not an Exception
            # This is uncommon but tests the non-Exception path
            raise CustomBaseException("Simulated base exception")

        # CustomBaseException should propagate through the exception handling
        with pytest.raises(CustomBaseException, match="Simulated base exception"):
            await processor.process(["test"], processor_with_base_exception)
