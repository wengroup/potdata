"""
Operations to manipulate data objects.
"""
import copy
import warnings
from typing import Any, Iterable

from monty.json import MSONable


class serializable_slice(MSONable):
    """
    A wrapper class around `slice` to make it serializable.

    See Python documentation for more information about `slice`.

    The difference is that you need to call the `to_slice` method to convert it to a
    `slice` object. After this, using it as a regular `slice` object.
    """

    def __init__(self, *args):
        self.args = args

    def to_slice(self) -> slice:
        return slice(*self.args)

    def as_dict(self):
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "args": self.args,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(*d["args"])


def slice_sequence(
    data: Iterable[Any], slicer: int | list[int] | serializable_slice | slice | None
) -> tuple[list[Any], list[int]]:
    """
    Slice a list of data.

    Args:
        data: the data to slice.
        slicer: indices of the data to select or a slice object.
            1. If an integer is provided, the index corresponding to the integer is
            selected. For example, `slicer=-1` will select the last data point.
            2. If a list of integers is provided, the corresponding indices are
            selected. For example, `slicer=[0, 3, 5]` will select data points with
            indices 0, 3, and 5.
            3. If a python slice object (or the serializable_slice) is provided,
            will perform the selection according to the slice. For example,
            `slicer=slice(0, None, 2)` will select data points with indices 0, 2, 4,...
            4. None is equivalent to `slice(None)`, i.e. select all data points.

    Returns:
        selected: Selected subset of data.
        indices: Indices of the selected data:w
    """
    data = [x for x in data]
    size = len(data)

    if isinstance(slicer, int):
        indices = [slicer]

    elif isinstance(slicer, (slice, serializable_slice)):
        if isinstance(slicer, serializable_slice):
            slicer = slicer.to_slice()
        start, stop, step = slicer.indices(size)
        indices = list(range(start, stop, step))

    elif isinstance(slicer, (list, tuple)):
        indices = [i for i in slicer if i < size]
        if max(slicer) >= size:
            larger = [i for i in slicer if i >= size]
            warnings.warn(
                f"Frame indices {larger} provided in slicer larger than the "
                f"number of total frames ({size}). They are ignored."
            )

    elif slicer is None:
        indices = list(range(size))

    else:
        supported = ("int", "list", "tuple", "slice", "None")
        raise RuntimeError(
            f"Expect `type(slicer)` be one of {supported}; got {type(slicer)}."
        )

    selected = [data[i] for i in indices]

    return selected, indices


def merge_dict(dct: dict, merge_dct: dict, new: bool = True) -> dict:
    """Recursive dict merge.

    Inspired by :meth:``dict.update()``, instead of updating only top-level keys,
    this function goes down into dicts nested to an arbitrary depth, updating keys.

    Args:
        dct: dict onto which the merge is executed.
        merge_dct: dct merged into dct.
        new: Whether to return a new dict or modify the original dict.

    Returns:
        dct: updated dict.
    """
    if new:
        dct = copy.deepcopy(dct)

    for k, v in merge_dct.items():
        if k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], dict):
            merge_dict(dct[k], merge_dct[k], new=False)
        else:
            dct[k] = merge_dct[k]

    return dct


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


def set_field_precision(obj, fields: list[str], digits: int = 10, new_obj: bool = True):
    """
    Set given fields of an object to a certain precision in scientific notation.

    Args:
        obj: The object to change the precision.
        fields: The fields to change the precision.
        digits: The precision, given by the number of digits after the decimal point.
        new_obj: Whether to return a new object or modify the original object.
    """

    fmt = "{" + f":.{digits}e" + "}"

    def _set_precision(x):
        if isinstance(x, float):
            return float(fmt.format(x))
        elif isinstance(x, Iterable):
            return [_set_precision(i) for i in x]
        else:
            return x

    if new_obj:
        obj = copy.deepcopy(obj)

    for field in fields:
        setattr(obj, field, _set_precision(getattr(obj, field)))

    return obj


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
