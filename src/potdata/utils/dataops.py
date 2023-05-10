"""
Operations to manipulate data objects.
"""
import copy
import warnings
from typing import Any, Iterable


def slice_sequence(
    data: Iterable[Any], slicer: int | list[int] | slice | None
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
            3. If a python slice object is provided, will perform the selection
            according to the slice. For example, `slicer=slice(0, None, 2)` will select
            data points with indices 0, 2, 4,...
            4. None is equivalent to `slice(None)`, i.e. select all data points.

    Returns:
        selected: Selected subset of data.
        indices: Indices of the selected data:w
    """
    data = [x for x in data]
    size = len(data)

    if isinstance(slicer, int):
        indices = [slicer]
    elif isinstance(slicer, slice):
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
    this function recurses down into dicts nested to an arbitrary depth, updating keys.

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
