def assert_preserves_label(transform, video):
    class my_label:
        pass

    frames, transformed_label = transform(video, my_label)
    assert transformed_label == my_label
