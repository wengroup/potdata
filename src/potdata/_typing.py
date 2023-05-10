import pathlib
from typing import Union

Vector3D = tuple[float, float, float]

Matrix3D = tuple[Vector3D, Vector3D, Vector3D]

Vector6D = tuple[float, float, float, float, float, float]
Matrix6D = tuple[Vector6D, Vector6D, Vector6D, Vector6D, Vector6D, Vector6D]

PathLike = Union[str, pathlib.Path]
