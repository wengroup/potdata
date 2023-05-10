import os
from pathlib import Path

from potdata._typing import PathLike


def to_path(path: PathLike) -> Path:
    """Convert a PathLike (i.e. str or pathlib.Path) to pathlib.Path."""
    return Path(path).expanduser().resolve()


def create_directory(path: PathLike, path_is_file: bool = False) -> Path:
    """
    Create a directory at the given path.

    Args:
        path: path to the directory or path to a file that is in the directory. If
            `path_is_file=False`, this function will create a directory specified by
            `path`. If `path_is_file=True`, this function will create a directory
            given by `path.parent`.
        path_is_file: where the given path is a file.

    Returns:
        Path to the created directory.
    """
    p = to_path(path)

    if path_is_file:
        dirname = p.parent
    else:
        dirname = p

    if not dirname.exists():
        os.makedirs(dirname)

    return dirname
