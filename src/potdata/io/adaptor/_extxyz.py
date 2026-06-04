"""VASP data adaptor."""

from typing import Any

import numpy as np
from monty.io import zopen
from pymatgen.core import Lattice, Structure

from potdata._typing import Matrix3D, PathLike, Vector3D
from potdata.schema.datapoint import DataCollection, DataPoint, Property
from potdata.utils.path import create_directory, to_path

from .base import BaseDataCollectionAdaptor, BaseDataPointAdaptor


class ExtxyzAdaptor(BaseDataPointAdaptor):
    """Extended xyz format adaptor."""

    def read(self, path: PathLike, energy_key: str = "Energy") -> DataPoint:
        """Read into a data point.

        Args:
            path: filename from which to read a DataPoint.
            energy_key: The key to the energy in the extended xyz file.
        """

        with zopen(path, "r") as fin:
            lines = fin.read()

        return self.from_string(lines, str(path), energy_key=energy_key)

    def write(
        self,
        datapoint: DataPoint,
        path: PathLike,
        *,
        reference_energy: dict[str, float] = None,
        mode: str = "w",
        stress_format: str = "full",
    ):
        """
        Write the data point to file.

        Args:
            datapoint: a DataPoint to convert.
            path: filename to write the DataPoint.
            reference_energy: A dictionary of reference energies for each species.
                In general, one would prefer to reference energy against the free atom
                energies. If `None`, the reference energy is set to zero.
            mode: mode to write to the file, e.g. `w` for writing and `a` for appending.
            stress_format: format of the stress tensor. Options are "full" or "voigt".
                If `full`, the 9 components of the stress tensor are provided. If
                `voigt`, the 6 components of the stress tensor s11, s22, s33, s23, s13,
                s12 are provided.
        """
        structure = datapoint.structure
        prop = datapoint.property

        s = self.to_string(
            cell=structure.lattice.matrix,
            species=[s.symbol for s in structure.species],
            coords=structure.cart_coords,
            pbc=structure.pbc,
            energy=datapoint.get_cohesive_energy(reference_energy=reference_energy),
            forces=prop.forces,
            stress=prop.stress,
            stress_format=stress_format,
        )
        with open(path, mode=mode) as f:
            f.write(s)

    @staticmethod
    def from_string(
        config: str,
        path: str = None,
        energy_key: str = "Energy",
    ) -> DataPoint:
        """Read xyz config from a string."""

        lines = config.splitlines()

        try:
            natoms = int(lines[0].split()[0])
        except ValueError as e:
            raise ValueError(f"{e}.\nCorrupted extxyz file {path} at line 1.")

        # lattice vector
        line1 = lines[1].replace("'", '"')
        cell = _parse_key_value(line1, "Lattice", "float", 9, path)
        cell = np.reshape(cell, (3, 3)).tolist()

        # PBC
        pbc_str = _parse_key_value(line1, "PBC", "str", 3, path)
        try:
            # `1` or `0`?
            pbc = [int(s) for s in pbc_str]
            pbc = [bool(i) for i in pbc]
        except ValueError:
            # `T` or `F`?
            pbc_str = [s.lower() for s in pbc_str]
            if not all([s in ["t", "f"] for s in pbc_str]):
                raise ValueError('PBC must be "T" or "F", or "1" or "0".')
            else:
                pbc = [True if s == "t" else False for s in pbc_str]

        # energy is optional
        try:
            in_quotes = _check_in_quotes(line1, energy_key, path)
            energy = _parse_key_value(line1, energy_key, "float", 1, path, in_quotes)[0]
        except RuntimeError:
            energy = None

        # stress is optional
        try:
            stress = _parse_key_value(line1, "Stress", "float", 9, path)
            stress = np.reshape(stress, (3, 3)).tolist()
        except RuntimeError:
            stress = None

        # body, species symbol, x, y, z (and fx, fy, fz if provided)
        species = []
        coords = []
        forces = []

        # if forces provided
        line2 = lines[2].strip().split()
        if len(line2) == 4:
            has_forces = False
        elif len(line2) == 7:
            has_forces = True
        else:
            raise ValueError(f"Corrupted data at line 3 of file {path}.")

        try:
            num_lines = 0
            for ln in lines[2:]:
                num_lines += 1
                line = ln.strip().split()
                if len(line) != 4 and len(line) != 7:
                    raise ValueError(
                        f"Corrupted data at line {num_lines + 3} of file `{path}`."
                    )
                if has_forces:
                    symbol, x, y, z, fx, fy, fz = line
                    species.append(symbol.lower().capitalize())
                    coords.append((float(x), float(y), float(z)))
                    forces.append((float(fx), float(fy), float(fz)))
                else:
                    symbol, x, y, z = line
                    species.append(symbol.lower().capitalize())
                    coords.append((float(x), float(y), float(z)))
        except ValueError as e:
            raise ValueError(
                f"{e}.\nCorrupted data at line {num_lines + 3} of file {path}."
            )

        if num_lines != natoms:
            raise RuntimeError(
                f"Corrupted data file {path}. Number of atoms is {natoms}, "
                f"whereas number of data lines is {num_lines}."
            )

        if not has_forces:
            forces = None

        structure = Structure(
            lattice=Lattice(cell, pbc),
            species=species,
            coords=coords,
            coords_are_cartesian=True,
        )
        prop = Property(energy=energy, forces=forces, stress=stress)
        datapoint = DataPoint(structure=structure, property=prop)

        return datapoint

    @staticmethod
    def to_string(
        cell: Matrix3D,
        species: list[str],
        coords: list[Vector3D],
        pbc: tuple[bool, bool, bool] | tuple[int, int, int],
        energy: float | None = None,
        forces: list[Vector3D] | None = None,
        stress: Matrix3D | None = None,
        stress_format: str = "full",
    ) -> str:
        """
        Convert the data to extxyz format as a string.

        Args:
            cell: supercell lattice vectors
            species: species of atoms
            coords: coordinates of atoms
            pbc: periodic boundary conditions
            energy: potential energy of the configuration.
            forces: Nx3 array, forces on atoms.
            stress: stress on the cell.
            stress_format: format of the stress tensor. Options are "full" or "voigt".
                If `full`, the 9 components of the stress tensor are provided. If
                `voigt`, the 6 components of the stress tensor s11, s22, s33, s23, s13,
                s12 are provided.
        Returns:
            Extxyz as a string.
        """
        s = ""

        # first line (number of atoms)
        natoms = len(species)
        s += f"{natoms}\n"

        # second line
        s += 'Lattice="'
        for i, row in enumerate(cell):
            for j, v in enumerate(row):
                s += f"{v:.15g} "
                if i == 2 and j == 2:
                    s = s[:-1] + '" '

        s += f'PBC="{int(pbc[0])} {int(pbc[1])} {int(pbc[2])}" '

        if energy is not None:
            s += f'Energy="{energy:.15g}" '

        if stress is not None:
            s += 'Stress="'

            if stress_format == "full":
                for i, row in enumerate(stress):
                    for j, v in enumerate(row):
                        s += f"{v:.15g} "
            elif stress_format == "voigt":
                s += f"{stress[0][0]:.15g} {stress[1][1]:.15g} {stress[2][2]:.15g} "
                s += f"{stress[1][2]:.15g} {stress[0][2]:.15g} {stress[0][1]:.15g} "
            else:
                supported = ("full", "voigt")
                raise ValueError(
                    f"Unknown stress format `{stress_format}`. Support are {supported}."
                )
            s = s[:-1] + '" '

        properties = "Properties=species:S:1:pos:R:3"
        if forces is not None:
            properties += ":for:R:3\n"
        else:
            properties += "\n"
        s += properties

        # body
        for i in range(natoms):
            s += f"{species[i]:2s} "
            s += f"{coords[i][0]:23.15e} {coords[i][1]:23.15e} {coords[i][2]:23.15e} "

            if forces is not None:
                s += (
                    f"{forces[i][0]:23.15e} {forces[i][1]:23.15e} {forces[i][2]:23.15e}"
                )

            s += "\n"

        return s


class ExtxyzCollectionAdaptor(BaseDataCollectionAdaptor):
    def read(self, path: PathLike, extension: str = ".xyz") -> DataCollection:  # type: ignore[override]
        """
        Read the data points from extxyz file(s).

        This will read all files in the directory and subdirectories with the extension.

        Args:
            path: path to a directory to hold the file(s) or path to a file with all
                the extended xyz configurations concatenated.
            extension: all files with the extension in `path` and its subdirectories
                will be read.

        Returns:
            A list of data points, the `label` attribute of each data point is set to
            path to the file.
        """

        adaptor = ExtxyzAdaptor()

        path = to_path(path)

        # read from a directory, assuming one config per file
        if path.is_dir():
            datapoints = []
            for p in path.rglob("*" + extension):
                if p.is_file():
                    dp = adaptor.read(p)
                    dp.label = p.as_posix()
                    datapoints.append(dp)

        # read from a single file; can consist of be multiple configs
        elif path.is_file():
            configs = self._separate_configs(path)
            datapoints = []

            starting_line = 0
            for i, c in enumerate(configs):
                dp = adaptor.from_string(
                    c, path=path.as_posix() + f". Starting line: {starting_line}"
                )
                starting_line += len(c.splitlines())
                dp.label = path.as_posix() + f" :config {i}"
                datapoints.append(dp)

        else:
            raise RuntimeError(f"Path `{path}` is not a file or directory.")

        dc = DataCollection(data_points=datapoints)

        return dc

    def write(
        self,
        data: DataCollection,
        path: PathLike,
        *,
        reference_energy: dict[str, float] = None,
        separate: bool = True,
        stress_format: str = "full",
    ) -> list[PathLike]:
        """
        Write the data points to extxyz file(s).

        Args:
            data: data points to write.
            path: path to a directory to hold the files.
            reference_energy: A dictionary of reference energies for each species.
                In general, one would prefer to reference energy against the free atom
                energies. If `None`, the reference energy is set to zero.
            separate: Whether to write to separate files: one for each data point.
                When `separate=False`, all data points are written to a single file
                given by `path`. When `separate=True`, one file for each data
                point, and it's up to the specific adaptor to determine the names of
                the files. In this case `path` is typically a directory and the files
                are written into it. For example, when `separate=True` and
                `path=/home/data`, a specific adaptor may write the files as
                `/home/data/datafile-1.xyz`, `/home/data/datafile-2.xyz`...
            stress_format: format of the stress tensor. Options are "full" or "voigt".
                If `full`, the 9 components of the stress tensor are provided. If
                `voigt`, the 6 components of the stress tensor s11, s22, s33, s23, s13,
                s12 are provided.
        """

        adaptor = ExtxyzAdaptor()

        datapoints = data.data_points

        if not separate:
            for dp in datapoints:
                adaptor.write(
                    dp,
                    path,
                    reference_energy=reference_energy,
                    mode="a",
                    stress_format=stress_format,
                )
            filenames = [path]

        else:
            directory = create_directory(path)
            filenames = [
                directory.joinpath(f"datapoint-{i:010d}.xyz")
                for i in range(len(datapoints))
            ]
            for f, dp in zip(filenames, datapoints):
                adaptor.write(
                    dp,
                    f,
                    reference_energy=reference_energy,
                    mode="w",
                    stress_format=stress_format,
                )

        return filenames

    @staticmethod
    def _separate_configs(path: PathLike) -> list[str]:
        """
        Separate multiple configurations in a single file.

        Returns:
            A list of strings, each string is a configuration.
        """
        with zopen(path, "r") as f:
            lines = f.readlines()

        configs = []
        i = 0
        while i < len(lines):
            num_atoms = int(lines[i])
            configs.append("".join(lines[i : i + num_atoms + 2]))
            i += num_atoms + 2

        return configs


def _parse_key_value(
    line: str,
    key: str,
    dtype: str,
    size: int,
    filename: PathLike,
    in_quotes: bool = True,
) -> list[Any]:
    """
    Given key, parse a string like ``other stuff key="value" other stuff`` to get value.

    If there is no space in value, the quotes `"` can be omitted.

    Args:
        line: The string line.
        key: Keyword to parse.
        dtype: Expected data type of value, `int`, `float`, or `str`.
        size: Expected size of value.
        filename: File name where the line comes from.
    Returns:
        Values associated with key.
    """
    line = line.strip()
    key = _check_key(line, key, filename)
    try:
        value = line[line.index(key) :]
        if in_quotes:
            value = value[value.index('"') + 1 :]
            value = value[: value.index('"')]
        else:
            value = value[value.index("=") + 1 :]
            value = value.lstrip(" ")
            value += " "  # add a whitespace at end in case this is the last key
            value = value[: value.index(" ")]

        value_list = value.split()
    except Exception as e:
        raise RuntimeError(f"{e}.\nCorrupted {key} data at line 2 of file {filename}.")

    if len(value_list) != size:
        raise RuntimeError(
            f"Incorrect size of {key} at line 2 of file {filename};\n"
            f"required: {size}, provided: {len(value_list)}. Possibly, the quotes not "
            f"match."
        )

    try:
        if dtype == "float":
            return [float(i) for i in value_list]
        elif dtype == "int":
            return [int(i) for i in value_list]
        elif dtype == "str":
            return [str(i) for i in value_list]
        else:
            raise ValueError(f"Unknown data type {dtype}.")
    except Exception as e:
        raise RuntimeError(f"{e}.\nCorrupted {key} data at line 2 of file {filename}.")


def _check_key(line, key, filename):
    """
    Check whether a key or its lowercase counterpart is in line.
    """
    if key not in line:
        key_lower = key.lower()
        if key_lower not in line:
            raise RuntimeError(f"{key} not found at line 2 of file {filename}.")
        else:
            key = key_lower
    return key


def _check_in_quotes(line, key, filename):
    """
    Check whether ``key=value`` or ``key="value"`` in line.
    """
    key = _check_key(line, key, filename)
    value = line[line.index(key) :]
    value = value[value.index("=") + 1 :]
    value = value.lstrip(" ")
    if value[0] == '"':
        return True
    else:
        return False
