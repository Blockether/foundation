"""Comprehensive performance benchmark for GraphDatabase with 50k entities."""

from __future__ import annotations

import time
from typing import TypedDict

import pytest

from blockether_foundation.graph import Entity, GraphDatabase


class InsertionResults(TypedDict):
    """Results from batch insertion benchmark."""

    entities_processed: int
    total_time: float
    batch_size: int
    total_batches: int
    insertion_rate: float
    avg_batch_time: float
    min_batch_time: float
    max_batch_time: float
    avg_entities_per_batch: float
    throughput_mbps: float


class SearchResults(TypedDict):
    """Results from search performance benchmark."""

    search_terms_tested: int
    search_times: list[float]
    avg_search_time: float
    max_search_time: float
    min_search_time: float
    searches_per_second: float
    total_results: int
    avg_results_per_search: float


class MemoryResults(TypedDict):
    """Results from memory and functionality validation."""

    entity_retrieval_first: bool
    entity_retrieval_mid: bool
    entity_retrieval_last: bool
    general_search_results: bool
    specific_search_results: bool
    entity_count_correct: bool
    expected_count: int
    actual_count: int
    database_functional: bool


class SearchBenchmarkResult(TypedDict):
    """Result from a single search benchmark."""

    term: str
    time: float
    count: int
    avg_score: float


class BenchmarkConfig(TypedDict):
    """Benchmark configuration."""

    total_entities: int
    batch_size: int
    min_rate: int


class TestGraphDatabaseBenchmark:
    """Comprehensive performance benchmark for GraphDatabase."""

    def __init__(self) -> None:
        """Initialize the test class with benchmark results storage."""
        self._benchmark_results: (
            dict[str, InsertionResults | SearchResults | MemoryResults | BenchmarkConfig] | None
        ) = None

    @pytest.mark.integration
    def test_comprehensive_50k_entities_benchmark(self) -> None:
        """Comprehensive benchmark: 50,000 entities in 20-element batches."""
        db = GraphDatabase()

        # Benchmark configuration constants
        TOTAL_ENTITIES = 50000
        BATCH_SIZE = 20
        MIN_INSERTION_RATE = (
            500  # entities per second - realistic expectation for batch performance
        )
        MAX_SEARCH_TIME = 0.5  # seconds - faster expectation for batch-loaded data
        SEARCH_TOP_K = 100
        ENTITY_TYPES = ["concept", "tool", "library", "framework", "algorithm", "methodology"]
        CONTENT_TEMPLATES = [
            "Machine learning {} for data processing and analysis",
            "Advanced {} implementation with optimization techniques",
            "Modern {} approach for scalable computing solutions",
            "Efficient {} algorithm for large dataset processing",
            "Cutting-edge {} methodology for real-time applications",
        ]
        SEARCH_TERMS = ["machine learning", "algorithm", "optimization", "scalable", "efficient"]
        MIN_INSERTION_TIME = 0.1  # seconds for average batch time
        MIN_ENTITY_COUNT = 50000
        GENERAL_SEARCH_TOP_K = 50
        SPECIFIC_SEARCH_TOP_K = 10
        MIN_OVERALL_THROUGHPUT = 300

        # Generate test entities using accumulation pattern
        entity_indices = list(range(TOTAL_ENTITIES))
        entity_type_list = [ENTITY_TYPES[i % len(ENTITY_TYPES)] for i in entity_indices]
        content_template_list = [
            CONTENT_TEMPLATES[i % len(CONTENT_TEMPLATES)] for i in entity_indices
        ]

        entities = [
            Entity(
                name=f"BenchmarkEntity {i}",
                type=entity_type_list[i],  # type: ignore
                content=content_template_list[i].format(entity_type_list[i]),
            )
            for i in entity_indices
        ]

        # Main insertion benchmark
        insertion_results: InsertionResults = self._benchmark_batch_insertion(
            db, entities, BATCH_SIZE
        )

        # Search performance benchmark
        search_results: SearchResults = self._benchmark_search_performance(
            db, SEARCH_TERMS, MAX_SEARCH_TIME, SEARCH_TOP_K
        )

        # Memory validation
        memory_results: MemoryResults = self._validate_database_functionality(
            db, MIN_ENTITY_COUNT, GENERAL_SEARCH_TOP_K, SPECIFIC_SEARCH_TOP_K
        )

        # Comprehensive performance validation
        self._validate_comprehensive_performance(
            insertion_results,
            search_results,
            memory_results,
            MIN_INSERTION_RATE,
            MIN_INSERTION_TIME,
            MIN_OVERALL_THROUGHPUT,
        )

        # Performance results summary
        self._print_performance_summary(
            insertion_results,
            search_results,
            memory_results,
            {
                "total_entities": TOTAL_ENTITIES,
                "batch_size": BATCH_SIZE,
                "min_rate": MIN_INSERTION_RATE,
            },
        )

        # Final comprehensive assertions
        assert insertion_results["entities_processed"] == TOTAL_ENTITIES, (
            f"Should process all {TOTAL_ENTITIES} entities"
        )
        assert search_results["search_terms_tested"] == len(SEARCH_TERMS), (
            f"Should test all {len(SEARCH_TERMS)} search terms"
        )
        assert memory_results["database_functional"], "Database should remain fully functional"
        assert insertion_results["insertion_rate"] >= MIN_INSERTION_RATE, (
            f"Insertion rate should meet minimum of {MIN_INSERTION_RATE} entities/sec"
        )

    def _print_performance_summary(
        self,
        insertion: InsertionResults,
        search: SearchResults,
        memory: MemoryResults,
        config: BenchmarkConfig,
    ) -> None:
        """Print comprehensive performance results summary."""
        # Store results for test completion
        self._benchmark_results = {
            "insertion": insertion,
            "search": search,
            "memory": memory,
            "config": config,
        }

        # Validate that results are properly stored
        assert self._benchmark_results is not None, "Benchmark results should be stored"
        assert "insertion" in self._benchmark_results, "Insertion results should be stored"
        assert "search" in self._benchmark_results, "Search results should be stored"
        assert "memory" in self._benchmark_results, "Memory results should be stored"
        assert "config" in self._benchmark_results, "Config should be stored"

    def _benchmark_batch_insertion(
        self, db: GraphDatabase, entities: list[Entity], batch_size: int
    ) -> InsertionResults:
        """Benchmark batch insertion performance."""
        ENTITY_SIZE_BYTES = 200  # Rough estimate for throughput calculation
        BYTES_PER_MB = 1024 * 1024

        start_time = time.time()
        batch_times = []
        entities_processed = 0

        # Create batch ranges using comprehension with accumulation pattern
        batch_indices = list(range(0, len(entities), batch_size))
        batch_ranges = [(i, min(i + batch_size, len(entities))) for i in batch_indices]

        # Process batches using accumulation pattern
        batch_times = [
            self._process_single_batch(db, entities[start_idx:end_idx])
            for start_idx, end_idx in batch_ranges
        ]

        entities_processed = len(entities)
        total_time = time.time() - start_time
        total_batches = len(batch_times)

        # Calculate performance metrics
        insertion_rate = entities_processed / total_time
        avg_batch_time = sum(batch_times) / total_batches
        avg_entities_per_batch = entities_processed / total_batches

        return {
            "entities_processed": entities_processed,
            "total_time": total_time,
            "batch_size": batch_size,
            "total_batches": total_batches,
            "insertion_rate": insertion_rate,
            "avg_batch_time": avg_batch_time,
            "min_batch_time": min(batch_times),
            "max_batch_time": max(batch_times),
            "avg_entities_per_batch": avg_entities_per_batch,
            "throughput_mbps": (entities_processed * ENTITY_SIZE_BYTES)
            / (total_time * BYTES_PER_MB),
        }

    def _process_single_batch(self, db: GraphDatabase, batch: list[Entity]) -> float:
        """Process a single batch and return the time taken."""
        batch_start = time.time()
        db.add_entities(batch)
        return time.time() - batch_start

    def _benchmark_search_performance(
        self, db: GraphDatabase, search_terms: list[str], max_search_time: float, search_top_k: int
    ) -> SearchResults:
        """Benchmark search performance after insertion."""

        # Perform searches and measure times using accumulation pattern
        search_results: list[SearchBenchmarkResult] = [
            self._perform_search_benchmark(db, term, search_top_k) for term in search_terms
        ]

        search_times: list[float] = [result["time"] for result in search_results]
        avg_search_time: float = sum(search_times) / len(search_times)
        max_actual_search_time: float = max(search_times)

        # Performance assertions
        assert max_actual_search_time < max_search_time, (
            f"Search too slow: {max_actual_search_time:.3f}s for single term, limit is {max_search_time:.3f}s"
        )
        assert all(result["count"] > 0 for result in search_results), (
            "All searches should return results"
        )

        return {
            "search_terms_tested": len(search_terms),
            "search_times": search_times,
            "avg_search_time": avg_search_time,
            "max_search_time": max_actual_search_time,
            "min_search_time": min(search_times),
            "searches_per_second": 1 / avg_search_time,
            "total_results": sum(result["count"] for result in search_results),
            "avg_results_per_search": sum(result["count"] for result in search_results)
            / len(search_results),
        }

    def _perform_search_benchmark(
        self, db: GraphDatabase, term: str, top_k: int
    ) -> SearchBenchmarkResult:
        """Perform a single search benchmark."""
        MIN_SCORE = 0.0  # Minimum valid score

        start_time = time.time()
        results = db.search(term, top_k=top_k)
        search_time = time.time() - start_time

        # Validate search results
        assert len(results) > 0, f"Search for '{term}' should return results"
        assert all(isinstance(score, float) and score > MIN_SCORE for _, score in results), (
            f"Search scores should be positive, above {MIN_SCORE}"
        )

        scores = [score for _, score in results]
        return {
            "term": term,
            "time": search_time,
            "count": len(results),
            "avg_score": sum(scores) / len(scores),
        }

    def _validate_database_functionality(
        self,
        db: GraphDatabase,
        expected_entity_count: int,
        general_search_top_k: int,
        specific_search_top_k: int,
    ) -> MemoryResults:
        """Validate database functionality after massive insertion."""
        FIRST_ENTITY_INDEX = 0
        MID_ENTITY_INDEX = 25000
        LAST_ENTITY_INDEX = 49999
        TEST_ENTITY_INDEX = 123
        GENERAL_SEARCH_TERM = "machine"

        # Test entity retrieval
        test_entity = db.get_entity_by_name(f"BenchmarkEntity {FIRST_ENTITY_INDEX}")
        entity_retrieval_works = test_entity is not None

        # Test mid-range entity
        mid_entity = db.get_entity_by_name(f"BenchmarkEntity {MID_ENTITY_INDEX}")
        mid_entity_retrieval_works = mid_entity is not None

        # Test last entity
        last_entity = db.get_entity_by_name(f"BenchmarkEntity {LAST_ENTITY_INDEX}")
        last_entity_retrieval_works = last_entity is not None

        # Test search functionality with different terms
        general_search = db.search(GENERAL_SEARCH_TERM, top_k=general_search_top_k)
        specific_search = db.search(
            f"BenchmarkEntity {TEST_ENTITY_INDEX}", top_k=specific_search_top_k
        )

        # Test entity count
        actual_count = db.entity_count

        functionality_flags = [
            entity_retrieval_works,
            mid_entity_retrieval_works,
            last_entity_retrieval_works,
            len(general_search) > 0,
            len(specific_search) > 0,
            actual_count == expected_entity_count,
        ]

        return {
            "entity_retrieval_first": entity_retrieval_works,
            "entity_retrieval_mid": mid_entity_retrieval_works,
            "entity_retrieval_last": last_entity_retrieval_works,
            "general_search_results": len(general_search) > 0,
            "specific_search_results": len(specific_search) > 0,
            "entity_count_correct": actual_count == expected_entity_count,
            "expected_count": expected_entity_count,
            "actual_count": actual_count,
            "database_functional": all(functionality_flags),
        }

    def _validate_comprehensive_performance(
        self,
        insertion: InsertionResults,
        search: SearchResults,
        memory: MemoryResults,
        min_insertion_rate: int,
        min_batch_time: float,
        min_overall_throughput: int,
    ) -> None:
        """Validate comprehensive performance meets expectations."""
        EXPECTED_ENTITIES_PROCESSED = 50000
        MAX_SEARCH_TIME_LIMIT = 1.0
        AVG_SEARCH_TIME_LIMIT = 0.5
        MIN_SEARCH_RATE = 2

        # Insertion performance validation
        assert insertion["entities_processed"] == EXPECTED_ENTITIES_PROCESSED, (
            f"Should process all {EXPECTED_ENTITIES_PROCESSED} entities, got {insertion['entities_processed']}"
        )
        assert insertion["insertion_rate"] > min_insertion_rate, (
            f"Insertion rate too slow: {insertion['insertion_rate']:.0f} entities/sec, minimum is {min_insertion_rate}"
        )
        assert insertion["avg_batch_time"] < min_batch_time, (
            f"Average batch time should be under {min_batch_time}s, was {insertion['avg_batch_time']:.3f}s"
        )

        # Search performance validation
        assert search["max_search_time"] < MAX_SEARCH_TIME_LIMIT, (
            f"Max search time too slow: {search['max_search_time']:.3f}s, limit is {MAX_SEARCH_TIME_LIMIT}s"
        )
        assert search["avg_search_time"] < AVG_SEARCH_TIME_LIMIT, (
            f"Average search time too slow: {search['avg_search_time']:.3f}s, limit is {AVG_SEARCH_TIME_LIMIT}s"
        )
        assert search["searches_per_second"] > MIN_SEARCH_RATE, (
            f"Search rate too slow: {search['searches_per_second']:.0f} searches/sec, minimum is {MIN_SEARCH_RATE}"
        )

        # Memory and functionality validation
        assert memory["database_functional"], "Database should remain fully functional"
        assert memory["entity_count_correct"], (
            f"Entity count mismatch: expected {memory['expected_count']}, got {memory['actual_count']}"
        )
        assert memory["general_search_results"] > 0, "General search should work"
        assert memory["specific_search_results"] > 0, "Specific search should work"

        # Overall performance validation
        total_time: float = insertion["total_time"]
        overall_throughput: float = insertion["entities_processed"] / total_time
        assert overall_throughput > min_overall_throughput, (
            f"Overall throughput too low: {overall_throughput:.0f} entities/sec, minimum is {min_overall_throughput}"
        )

        # Additional comprehensive validations
        assert insertion["total_batches"] > 0, "Should have processed at least one batch"
        assert search["total_results"] > 0, "Search should have returned some results"
        assert search["search_terms_tested"] > 0, "Should have tested at least one search term"
