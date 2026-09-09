from app.models.chunk import Chunk
from app.models.retrieval_result import RetrievalResult
from app.retrieval.reciprocal_rank_fusion import reciprocal_rank_fusion


def create_result(
    content: str,
    source: str,
    chunk_index: int,
    score: float = 1.0,
) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(
            content=content,
            source=source,
            file_type=".txt",
            chunk_index=chunk_index,
        ),
        retrieval_score=score,
    )


def test_returns_empty_list_for_empty_result_lists():
    results = reciprocal_rank_fusion([[], []])

    assert results == []


def test_preserves_order_for_single_result_list():
    result_a = create_result(
        content="Chunk A",
        source="test.txt",
        chunk_index=0,
    )
    result_b = create_result(
        content="Chunk B",
        source="test.txt",
        chunk_index=1,
    )
    result_c = create_result(
        content="Chunk C",
        source="test.txt",
        chunk_index=2,
    )

    results = reciprocal_rank_fusion([[result_a, result_b, result_c]])

    assert [result.chunk for result in results] == [
        result_a.chunk,
        result_b.chunk,
        result_c.chunk,
    ]


def test_rewards_chunks_that_appear_in_multiple_rankings():
    result_a = create_result(
        content="Chunk A",
        source="test.txt",
        chunk_index=0,
    )
    result_b = create_result(
        content="Chunk B",
        source="test.txt",
        chunk_index=1,
    )
    result_c = create_result(
        content="Chunk C",
        source="test.txt",
        chunk_index=2,
    )
    result_d = create_result(
        content="Chunk D",
        source="test.txt",
        chunk_index=3,
    )

    vector_results = [
        result_a,
        result_b,
        result_c,
    ]

    bm25_results = [
        result_c,
        result_a,
        result_d,
    ]

    results = reciprocal_rank_fusion([vector_results, bm25_results])

    ranked_chunks = [result.chunk for result in results]

    assert ranked_chunks[0] in [result_a.chunk, result_c.chunk]
    assert ranked_chunks[1] in [result_a.chunk, result_c.chunk]

    assert ranked_chunks.index(result_a.chunk) < ranked_chunks.index(result_b.chunk)
    assert ranked_chunks.index(result_c.chunk) < ranked_chunks.index(result_d.chunk)


def test_does_not_return_duplicate_chunks():
    result_a_vector = create_result(
        content="Chunk A",
        source="test.txt",
        chunk_index=0,
        score=0.9,
    )

    result_a_bm25 = create_result(
        content="Chunk A",
        source="test.txt",
        chunk_index=0,
        score=5.2,
    )

    results = reciprocal_rank_fusion(
        [
            [result_a_vector],
            [result_a_bm25],
        ]
    )

    assert len(results) == 1
    assert results[0].chunk == result_a_vector.chunk
