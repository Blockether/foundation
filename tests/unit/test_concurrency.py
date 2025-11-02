"""Tests for the ConcurrentProcessor class."""

import asyncio
import time
from collections.abc import Sequence

import pytest

from blockether_foundation.concurrency import ConcurrentProcessor


class TestConcurrentProcessor:
    """Test suite for ConcurrentProcessor class."""

    @pytest.mark.unit
    def test_processor_initialization_with_defaults(self):
        """Test processor initialization with default values."""
        processor = ConcurrentProcessor[str, str]()

        assert processor._concurrency == processor.DEFAULT_CONCURRENCY
        assert processor._max_retries == processor.DEFAULT_MAX_RETRIES
        assert processor._retry_min_wait == processor.DEFAULT_RETRY_MIN_WAIT
        assert processor._retry_max_wait == processor.DEFAULT_RETRY_MAX_WAIT
        assert processor._retry_exceptions == (Exception,)

    @pytest.mark.unit
    def test_processor_initialization_with_custom_values(self):
        """Test processor initialization with custom values."""
        processor = ConcurrentProcessor[int, str](
            concurrency=10,
            max_retries=5,
            retry_min_wait=1000,
            retry_max_wait=5000,
            retry_exceptions=(ValueError, TypeError),
        )

        assert processor._concurrency == 10
        assert processor._max_retries == 5
        assert processor._retry_min_wait == 1000
        assert processor._retry_max_wait == 5000
        assert processor._retry_exceptions == (ValueError, TypeError)

    @pytest.mark.unit
    def test_process_empty_items_list(self):
        """Test processing an empty list of items."""
        processor = ConcurrentProcessor[str, str]()

        async def mock_processor(item: str) -> Sequence[str]:
            return [f"processed: {item}"]

        result = asyncio.run(processor.process([], mock_processor))
        assert result == []

    @pytest.mark.unit
    def test_process_single_item(self):
        """Test processing a single item."""
        processor = ConcurrentProcessor[str, str]()

        async def mock_processor(item: str) -> Sequence[str]:
            return [f"processed: {item}"]

        result = asyncio.run(processor.process(["test"], mock_processor))
        assert result == ["processed: test"]

    @pytest.mark.unit
    def test_process_multiple_items_order_preservation(self):
        """Test that processing preserves input order."""
        processor = ConcurrentProcessor[str, str]()

        async def mock_processor(item: str) -> Sequence[str]:
            # Add delay to ensure concurrent processing
            await asyncio.sleep(0.1)
            return [f"processed: {item}"]

        items = ["item1", "item2", "item3"]
        result = asyncio.run(processor.process(items, mock_processor))
        assert result == ["processed: item1", "processed: item2", "processed: item3"]

    @pytest.mark.unit
    def test_process_concurrency_limits(self):
        """Test that concurrency limits are respected."""
        processor = ConcurrentProcessor[str, str](concurrency=2)

        # Track concurrent executions
        concurrent_count = 0
        max_concurrent = 0

        async def mock_processor(item: str) -> Sequence[str]:
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.1)
            concurrent_count -= 1
            return [f"processed: {item}"]

        items = ["item1", "item2", "item3", "item4"]
        result = asyncio.run(processor.process(items, mock_processor))

        assert len(result) == 4
        assert max_concurrent <= 2  # Should never exceed concurrency limit

    @pytest.mark.unit
    def test_process_with_list_return_values(self):
        """Test processing when processor returns a list."""
        processor = ConcurrentProcessor[str, str]()

        async def mock_processor(item: str) -> Sequence[str]:
            return [f"result1_{item}", f"result2_{item}"]

        result = asyncio.run(processor.process(["a", "b"], mock_processor))
        assert result == ["result1_a", "result2_a", "result1_b", "result2_b"]

    @pytest.mark.unit
    def test_process_with_single_return_value(self):
        """Test processing when processor returns a single value."""
        processor = ConcurrentProcessor[str, str]()

        async def mock_processor(item: str) -> Sequence[str]:
            return [f"processed: {item}"]

        result = asyncio.run(processor.process(["x", "y"], mock_processor))
        assert result == ["processed: x", "processed: y"]

    @pytest.mark.unit
    def test_process_with_string_return_value(self):
        """Test processing when processor returns a string."""
        processor = ConcurrentProcessor[str, str]()

        async def mock_processor(item: str) -> Sequence[str]:
            return [item.upper()]  # Return as a list with one item

        result = asyncio.run(processor.process(["hello", "world"], mock_processor))
        assert result == ["HELLO", "WORLD"]

    @pytest.mark.unit
    def test_process_with_none_return_value(self):
        """Test processing when processor returns None."""
        processor = ConcurrentProcessor[str, str]()

        async def mock_processor(item: str) -> Sequence[str]:
            return []  # Return empty list for None equivalent

        result = asyncio.run(processor.process(["a", "b"], mock_processor))
        assert result == []

    @pytest.mark.unit
    def test_process_with_none_in_list_return(self):
        """Test processing when processor returns list containing None."""
        processor = ConcurrentProcessor[str, str]()

        async def mock_processor(item: str) -> Sequence[str]:
            return [item, None, item]  # Include None in the list
            # This should be filtered out by flatten_results

        result = asyncio.run(processor.process(["x"], mock_processor))
        # None values should be filtered out
        assert result == ["x", "x"]

    @pytest.mark.unit
    def test_retry_logic_with_transient_failure(self):
        """Test retry logic for transient failures."""
        processor = ConcurrentProcessor[str, str](max_retries=3, retry_min_wait=10)

        call_count = 0

        async def mock_processor(item: str) -> Sequence[str]:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Fail first 2 times
                raise ConnectionError("Transient error")
            return [f"processed: {item}"]

        result = asyncio.run(processor.process(["test"], mock_processor))
        assert result == ["processed: test"]
        assert call_count == 3  # Should have retried twice

    @pytest.mark.unit
    def test_retry_logic_with_permanent_failure(self):
        """Test retry logic for permanent failures."""
        processor = ConcurrentProcessor[str, str](max_retries=2, retry_min_wait=10)

        async def mock_processor(item: str) -> Sequence[str]:
            raise ValueError("Permanent error")

        with pytest.raises(ValueError, match="Permanent error"):
            asyncio.run(processor.process(["test"], mock_processor))

    @pytest.mark.unit
    def test_retry_logic_with_custom_exception_types(self):
        """Test retry logic with custom exception types."""
        processor = ConcurrentProcessor[str, str](
            max_retries=2,
            retry_min_wait=10,
            retry_exceptions=(ConnectionError,),
        )

        async def mock_processor(item: str) -> Sequence[str]:
            raise ValueError("Non-retryable error")

        # Should fail immediately because ValueError is not in retry_exceptions
        with pytest.raises(ValueError, match="Non-retryable error"):
            asyncio.run(processor.process(["test"], mock_processor))

    @pytest.mark.unit
    def test_all_items_fail_atomically(self):
        """Test that all items fail atomically if any item fails permanently."""
        processor = ConcurrentProcessor[str, str](max_retries=1, retry_min_wait=10)

        async def mock_processor(item: str) -> Sequence[str]:
            if item == "fail":
                raise ValueError("Permanent failure")
            return [f"processed: {item}"]

        with pytest.raises(ValueError, match="Permanent failure"):
            asyncio.run(processor.process(["ok", "fail", "ok2"], mock_processor))

    @pytest.mark.unit
    def test_concurrent_execution_performance(self):
        """Test that concurrent execution provides performance benefits."""
        processor_fast = ConcurrentProcessor[str, str](concurrency=10)
        processor_slow = ConcurrentProcessor[str, str](concurrency=1)

        async def mock_processor(item: str) -> Sequence[str]:
            await asyncio.sleep(0.01)  # Small delay
            return [f"processed: {item}"]

        items = ["item1", "item2", "item3", "item4", "item5"]

        # Measure time for concurrent execution
        start_time = time.time()
        result_fast = asyncio.run(processor_fast.process(items, mock_processor))
        fast_time = time.time() - start_time

        # Measure time for sequential execution
        start_time = time.time()
        result_slow = asyncio.run(processor_slow.process(items, mock_processor))
        slow_time = time.time() - start_time

        # Results should be the same
        assert result_fast == result_slow

        # Concurrent should be faster (allow some tolerance)
        assert fast_time < slow_time * 0.8

    @pytest.mark.unit
    def test_processor_function_signature_variants(self):
        """Test processor function with different return type annotations."""
        processor = ConcurrentProcessor[str, int]()

        # Return List[str] which should be converted to Sequence[int]
        async def mock_processor_list(item: str) -> Sequence[int]:
            return [len(item)]

        # Return tuple which should be converted to Sequence[int]
        async def mock_processor_tuple(item: str) -> Sequence[int]:
            return (len(item),)

        result_list = asyncio.run(processor.process(["hello"], mock_processor_list))
        assert result_list == [5]

        result_tuple = asyncio.run(processor.process(["world"], mock_processor_tuple))
        assert result_tuple == [5]

    @pytest.mark.unit
    def test_processor_with_tuple_return(self):
        """Test processor function returning tuple."""
        processor = ConcurrentProcessor[str, str]()

        async def mock_processor(item: str) -> Sequence[str]:
            return (f"tuple_{item}",)  # Return as tuple

        result = asyncio.run(processor.process(["test"], mock_processor))
        assert result == ["tuple_test"]

    @pytest.mark.unit
    def test_base_exception_handling(self):
        """Test handling of BaseException subclasses."""
        processor = ConcurrentProcessor[str, str](max_retries=2, retry_min_wait=10)

        async def mock_processor(item: str) -> Sequence[str]:
            raise KeyboardInterrupt("Interrupted")

        # BaseException subclasses should not be retried
        with pytest.raises(KeyboardInterrupt, match="Interrupted"):
            asyncio.run(processor.process(["test"], mock_processor))