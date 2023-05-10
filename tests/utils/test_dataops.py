from potdata.utils.dataops import merge_dict


def test_merge_dict():
    d1 = {"a": 1, "b": {"b1": 2, "b2": 3}}
    d2 = {"a": 11, "b": {"b2": 33, "b3": 4}, "c": 5}
    ref = {"a": 11, "b": {"b1": 2, "b2": 33, "b3": 4}, "c": 5}

    merged = merge_dict(d1, d2)

    assert merged == ref
