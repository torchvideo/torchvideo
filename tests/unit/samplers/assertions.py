from typing import List

from assertions.seq import assert_ordered, assert_elems_lt, assert_elems_gte


def assert_valid_frame_index(
    frame_idx: List[int], expected_frame_count: int, video_length: int
):
    assert len(frame_idx) == expected_frame_count
    assert_ordered(frame_idx)
    assert_elems_lt(frame_idx, video_length)
    assert_elems_gte(frame_idx, 0)


def assert_valid_snippet_index(
    snippet_idx: List[int],
    expected_snippet_length: int,
    expected_segment_count: int,
    video_length: int,
):
    assert len(snippet_idx) == expected_segment_count * expected_snippet_length
    assert 0 < max(snippet_idx) < video_length
    current_min_index = snippet_idx[0]
    current_snippet_min_index = snippet_idx[0]

    for i in range(1, len(snippet_idx)):
        new_snippet = snippet_idx[i] < snippet_idx[i - 1]
        if new_snippet:
            assert snippet_idx[i] >= current_snippet_min_index
            current_snippet_min_index = snippet_idx[i]

        assert snippet_idx[i] >= current_min_index
        assert snippet_idx[i] >= current_snippet_min_index
