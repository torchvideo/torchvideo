def _is_int(maybe_int):
    try:
        return int(maybe_int) == maybe_int
    except TypeError:
        pass
    return False
