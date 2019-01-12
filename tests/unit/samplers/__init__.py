from assertions.seq import assert_ordered, assert_elems_lt, assert_elems_gte


def assert_valid_frame_index(frame_idx, expected_frame_count, video_length):
    assert len(frame_idx) == expected_frame_count
    assert_ordered(frame_idx)
    assert_elems_lt(frame_idx, video_length)
    assert_elems_gte(frame_idx, 0)
