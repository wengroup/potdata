import copy
from typing import Any, Iterable


def set_field_to_none(obj, fields: list[str], new_obj: bool = True):
    """
    Set given fields of an object to None.

    Args:
        obj: The object to set fields to None.
        fields: The fields to set to None.
        new_obj: Whether to return a new object or modify the original object.
    """
    if new_obj:
        obj = copy.deepcopy(obj)

    for field in fields:
        setattr(obj, field, None)

    return obj


def set_field_precision(obj, fields: list[str], digits: int = 10, new_obj: bool = True):
    """
    Set given fields of an object to a certain precision.

    Args:
        obj: The object to change the precision.
        fields: The fields to change the precision.
        digits: The precision, given by the number of digits after the decimal point.
        new_obj: Whether to return a new object or modify the original object.
    """

    def _set_precision(x):
        if isinstance(x, float):
            return round(x, digits)
        elif isinstance(x, Iterable):
            return [_set_precision(i) for i in x]
        else:
            return x

    if new_obj:
        obj = copy.deepcopy(obj)

    for field in fields:
        setattr(obj, field, _set_precision(getattr(obj, field)))

    return obj


def remove_none_from_dict(d: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively remove all keys with value None from a dictionary.

    Args:
        d: The dictionary to remove keys with value None.

    Returns:
        The dictionary with keys with value None removed.
    """
    new_d = {}

    for k, v in d.items():
        if v is not None:
            if isinstance(v, dict):
                new_d[k] = remove_none_from_dict(v)
            else:
                new_d[k] = v

    return new_d
