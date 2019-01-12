import numpy as np


def assert_ordered(seq):
    if len(seq) < 2:
        return
    else:
        prev_elem = seq[0]
        for i, elem in enumerate(seq[1:]):
            i += 1  # Index is offset by 1 since we pull first element off
            if prev_elem > elem:
                raise AssertionError(
                    "Expected {} > {} at index {}, {}".format(prev_elem, elem, i - 1, i)
                )
    return


def assert_elems_lt(seq, upper_bound):
    assert np.all(np.array(seq) < upper_bound)


def assert_elems_gte(seq, lower_bound):
    assert np.all(lower_bound <= np.array(seq))
