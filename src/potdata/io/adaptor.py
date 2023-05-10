"""Adaptors to convert a single DataPoint or a DataCollection to other format,
such as extended xyz files, DeepMD format, and ACE format.
"""
import copy
import itertools
import os
import random
from itertools import groupby
from typing import Any

import ase
import ase.io
import numpy as np
import pandas as pd
import ruamel.yaml
from jobflow import Maker, job
from monty.io import zopen
from monty.json import MSONable
from potdata._typing import Matrix3D, PathLike, Vector3D
from potdata.schema.datapoint import Configuration, DataCollection, DataPoint, Property
from potdata.utils.dataops import slice_sequence
from potdata.utils.misc import remove_none_from_dict
from potdata.utils.path import create_directory, to_path
from potdata.utils.units import kbar_to_eV_per_A_cube
from pymatgen.io.vasp import Vasprun
from scipy.spatial import distance_matrix

__all__ = [
    "BaseDataPointAdaptor",
    "BaseDataCollectionAdaptor",
    "VasprunAdaptor",
    "VasprunCollectionAdaptor",
    "ExtxyzAdaptor",
    "ExtxyzCollectionAdaptor",
    "YAMLCollectionAdaptor",
    "ACECollectionAdaptor",
    "MTPCollectionAdaptor",
    "DeepMDCollectionAdaptorOld",
]


class BaseDataPointAdaptor(MSONable):
    """
    Base adaptor that converts a DataPoint to other formats.

    Subclass should implement a `write()` method to convert a
    :obj:`~potdata.schema.datapoint.DataPoint` to other format, and it is optional
    for a subclass to implement a read function to convert other format to a DataPoint.
    """

    def read(self, path: PathLike) -> DataPoint | list[DataPoint]:
        """
        Read into a data point.

        Args:
            path: filename from which to read a DataPoint.
        """
        raise NotImplementedError

    def write(
        self,
        datapoint: DataPoint,
        path: PathLike,
        *,
        reference_energy: dict[str, float] = None,
        mode: str = "w",
    ) -> PathLike:
        """
        Write the data point to file.

        Args:
            datapoint: a DataPoint to convert.
            path: filename to write the DataPoint.
            reference_energy: A dictionary of reference energies for each species.
                In general, one would prefer to reference energy against the free atom
                energies. If `None`, the reference energy is set to zero.
            mode: mode to write to the file, e.g. `w` for writing and `a` for appending.

        Returns:
            Path to the file written.
        """
        raise NotImplementedError


class BaseDataCollectionAdaptor(MSONable):
    """
    Base adaptor that converts a DataCollection to other formats.

    Subclass of this class should implement a `write()` method to convert a
    :obj:`~potdata.schema.datapoint.DataCollection` to other format, and it is optional
    for a subclass to implement a read function to convert other format to a
    DataCollection.
    """

    def read(self, path: PathLike | list[PathLike]) -> DataCollection:
        """
        Read the data collection.

        Args: Path to read the data collection. This can be a path to a file or path
        to a directory.

        """
        raise NotImplementedError

    def write(
        self,
        data: DataCollection,
        path: PathLike,
        *,
        reference_energy: dict[str, float] = None,
    ) -> PathLike | list[PathLike]:
        """
        Write the data collection to file(s).

        Args:
            data: Data points to write.
            path: Path to write the data collection. This can be a path to a file or
                to a directory.
            reference_energy: A dictionary of reference energies for each species.
                In general, one would prefer to reference energy against the free atom
                energies. If `None`, the reference energy is set to zero.

        Returns:
            Path to the file or a list of filenames to which the data are written.
        """
        raise NotImplementedError


class VasprunAdaptor(BaseDataPointAdaptor):
    """VASP vasprun.xml adaptor."""

    def read(
        self, path: PathLike, index: int | list[int] | slice | None = -1
    ) -> list[DataPoint]:
        """Read vasprun.xml file into a list of data points.

        Args:
            path: Path to the vasprun.xml file.
            index: Index of the ionic step to read. Default to select the last ionic
                step. If `None`, all ionic steps are read. See
                :obj:`potdata.utils.dataops.slice_sequence` for more information on
                advanced selection.
        """
        vasprun = Vasprun(
            path,
            parse_dos=False,
            parse_eigen=False,
            parse_projected_eigen=False,
            parse_potcar_file=False,
        )
        ionic_steps, _ = slice_sequence(vasprun.ionic_steps, slicer=index)

        # units conversion from kbar to eV/A^3
        # VASP uses compression as the positive direction for stress, opposite to the
        # convention. Therefore, the sign is flipped with the minus sign.
        ratio = -kbar_to_eV_per_A_cube()

        datapoints = []
        for step in ionic_steps:
            dp = DataPoint(
                configuration=Configuration.from_pymatgen_structure(step["structure"]),
                property=Property(
                    forces=step["forces"],
                    stress=(ratio * np.asarray(step["stress"])).tolist(),
                    energy=step["e_0_energy"],
                ),
            )
            datapoints.append(dp)

        return datapoints


class VasprunCollectionAdaptor(BaseDataCollectionAdaptor):
    def read(  # type: ignore[override]
        self,
        path: PathLike,
        index: int | list[int] | slice | None = -1,
        name_pattern: str = "vasprun.xml",
    ) -> DataCollection:
        """Read all vasprun.xml from a directory into a list of data points.

        Args:
            path: List of paths to the vasprun.xml files.
            index: Index of the ionic step to read. Default to select the last ionic
                step. See `VasprunAdaptor.read()` for more information.
            name_pattern: All files with `<name_pattern>` in the filename will be
                treated as vasprun.xml files.
        """

        adaptor = VasprunAdaptor()

        datapoints = []
        for p in to_path(path).rglob(f"*{name_pattern}*"):
            if p.is_file():
                datapoints.extend(adaptor.read(p, index=index))

        dc = DataCollection(data_points=datapoints)

        return dc


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
    ):
        conf = datapoint.configuration
        prop = datapoint.property

        s = self.to_string(
            cell=conf.cell,
            species=conf.species,
            coords=conf.coords,
            pbc=conf.pbc,
            energy=datapoint.get_cohesive_energy(reference_energy=reference_energy),
            forces=prop.forces,
            stress=prop.stress,
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
        PBC_str = _parse_key_value(line1, "PBC", "str", 3, path)
        try:
            PBC = [int(s) for s in PBC_str]
        except ValueError:
            # `T` or `F`?
            PBC_str = [s.lower() for s in PBC_str]
            if not all([s in ["t", "f"] for s in PBC_str]):
                raise ValueError('PBC must be "T" or "F", or "1" or "0".')
            else:
                PBC = [1 if s == "t" else 0 for s in PBC_str]

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
                        f"Corrupted data at line {num_lines + 3} of file " f"`{path}`."
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

        conf = Configuration(species=species, coords=coords, cell=cell, pbc=PBC)
        prop = Property(energy=energy, forces=forces, stress=stress)
        datapoint = DataPoint(configuration=conf, property=prop)

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
            for i, row in enumerate(stress):
                for j, v in enumerate(row):
                    s += f"{v:.15g} "
                    if i == 2 and j == 2:
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

        # read from a directory
        if path.is_dir():
            datapoints = []
            for p in path.rglob("*" + extension):
                if p.is_file():
                    dp = adaptor.read(p)
                    dp.label = p.as_posix()
                    datapoints.append(dp)

        # read from a single file
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
                `/home/data/datafile-1.json`, `/home/data/datafile-2.json`...
        """

        adaptor = ExtxyzAdaptor()

        datapoints = data.data_points

        if not separate:
            for dp in datapoints:
                adaptor.write(dp, path, reference_energy=reference_energy, mode="a")
            filenames = [path]

        else:
            directory = create_directory(path)
            filenames = [
                directory.joinpath(f"datapoint-{i:010d}.xyz")
                for i in range(len(datapoints))
            ]
            for f, dp in zip(filenames, datapoints):
                adaptor.write(dp, f, reference_energy=reference_energy, mode="w")

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


class YAMLCollectionAdaptor(BaseDataCollectionAdaptor):
    def read(self, path: PathLike) -> DataCollection:  # type: ignore[override]
        """
        Read the data collection from a YAML file.

        It can be a list of DataPoints or a DataCollection.

        Args:
            path: path to the YAML file.

        Returns:
            A list of data points.
        """
        path = to_path(path)

        with zopen(path, "r") as f:
            data = ruamel.yaml.safe_load(f)

        if isinstance(data, (list, tuple)):
            datapoints = [DataPoint(**d) for d in data]
            dc = DataCollection(data_points=datapoints)
        else:
            dp = data.pop("data_points")
            datapoints = [DataPoint(**d) for d in dp]
            dc = DataCollection(data_points=datapoints, **data)

        return dc

    # TODO: add dealing with reference_energy
    def write(
        self,
        data: DataCollection,
        path: PathLike,
        *,
        reference_energy: dict[str, float] = None,
        as_list: bool = False,
    ) -> list[PathLike]:
        """
        Write the data points to a YAML file.

        Args:
            data: data points to write.
            path: path to the YAML file.
            reference_energy: A dictionary of reference energies for each species.
                In general, one would prefer to reference energy against the free atom
                energies. If `None`, the reference energy is set to zero.
            as_list: Whether to write the data as a list data points or as a data
                collection (which contains other metadata such as the label).
        """
        path = to_path(path)

        datapoints = [remove_none_from_dict(dp.dict()) for dp in data.data_points]

        if as_list:
            out = datapoints
        else:
            out = copy.copy(data)
            out.data_points = datapoints  # type: ignore
            out = out.dict()  # type: ignore

        with open(path, "w") as f:
            ruamel.yaml.safe_dump(out, f)

        return [path]


class ACECollectionAdaptor(BaseDataCollectionAdaptor):
    def read(  # type: ignore[override]
        self, path: PathLike, energy_column: str = "energy_corrected"
    ) -> DataCollection:
        """

        Args:
            path: path to the ACE data file.
            energy_column: column name in the dataframe to use as energy in the data
                points. Note, in ACE dataframe, the column `energy` is the raw energy
                and the column `energy_corrected` is the energy corrected by subtracting
                the reference energy of individual atoms.
        """

        # note, no stress is read
        def _get_dp(row):
            return DataPoint(
                configuration=Configuration.from_ase(row["ase_atoms"]),
                property=Property(energy=row["energy"], forces=row["forces"]),
            )

        df = pd.read_pickle(path, compression="gzip")
        datapoints = df.apply(_get_dp, axis=1).tolist()

        dc = DataCollection(data_points=datapoints)

        return dc

    def write(
        self,
        data: DataCollection,
        path: PathLike,
        *,
        reference_energy: dict[str, float] = None,
    ) -> list[PathLike]:
        """
        Write the data points to ACE format.

        It is a pickle file that contains a pandas dataframe.


        Args:
            data: data points to write.
            path: path to a directory to hold the files.
            reference_energy: A dictionary of reference energies for each species.
                In general, one would prefer to reference energy against the free atom
                energies. If `None`, the reference energy is set to zero.
        """

        datapoints = data.data_points

        df = pd.DataFrame(
            {
                "energy": [dp.property.energy for dp in datapoints],
                "forces": [dp.property.forces for dp in datapoints],
                "ase_atoms": [
                    ase.Atoms(
                        symbols=dp.configuration.species,
                        positions=dp.configuration.coords,
                        cell=dp.configuration.cell,
                        pbc=dp.configuration.pbc,
                    )
                    for dp in datapoints
                ],
                "energy_corrected": [
                    dp.get_cohesive_energy(reference_energy=reference_energy)
                    for dp in datapoints
                ],
            }
        )

        path = to_path(path)
        if path.suffix not in [".gz", ".gzip"]:
            path = to_path(path.as_posix() + ".gz")

        df.to_pickle(path, compression="gzip", protocol=4)

        return [path]


class MTPCollectionAdaptor(BaseDataCollectionAdaptor):
    def write(
        self,
        data: DataCollection,
        path: PathLike,
        *,
        reference_energy: dict[str, float] = None,
    ) -> PathLike:
        """
        Write the data points to MTP cfg format.

        Args:
            data: data points to write.
            path: path to the file written.
            reference_energy: A dictionary of reference energies for each species.
                In general, one would prefer to reference energy against the free atom
                energies. If `None`, the reference energy is set to zero.

        Returns: path to the file written.
        """

        species_map = data.get_species_mapping()

        s = ""
        for dp in data.data_points:
            s += self.as_string(dp, species_map, reference_energy) + "\n"

        path = to_path(path)
        with open(path, "w") as f:
            f.write(s)

        return path

    def as_string(
        self,
        dp: DataPoint,
        species_map: dict[str, int],
        reference_energy: dict[str, float] = None,
    ) -> str:
        """
        Convert a data point to a string in MTP cfg format.

        Args:
            dp: data point to convert.
            species_map: A dictionary of species string to species integer.
            reference_energy: A dictionary of reference energies for each species.

        Returns:
            A configuration in MTP cfg format as a string.
        """
        coords = dp.configuration.coords
        size = len(coords)
        energy = dp.get_cohesive_energy(reference_energy=reference_energy)
        min_dist = self._get_min_dist(coords)

        s = "BEGIN_CFG\n"
        s += " Size\n"
        s += f"{size:>5}\n"

        s += " Supercell\n"
        cell = dp.configuration.cell
        for line in cell:
            for item in line:
                s += f" {item:>15.6f}"
            s += "\n"

        # specie, coords and forces
        fmt = "{:>14s}{:>5s}{:>14s}{:>14s}{:>14s}{:>14s}{:>14s}{:>14s}{:>14s}\n"
        s += fmt.format(
            "AtomData:",
            "id",
            "type",
            "cartes_x",
            "cartes_y",
            "cartes_z",
            "fx",
            "fy",
            "fz",
        )

        for i, (sp, co, fo) in enumerate(
            zip(dp.configuration.species, coords, dp.property.forces)
        ):
            fmt = (
                "{:>19d}{:>14d}{:>14.6f}{:>14.6f}{:>14.6f}{:>14.6f}{:>14.6f}{:>14.6f}\n"
            )
            s += fmt.format(i + 1, species_map[sp], *co, *fo)

        # energy
        s += " Energy\n"
        s += f"     {energy:.12f}\n"

        # stress
        vs = stress_to_virial(dp.property.stress, dp.configuration.cell, sign=-1)
        fmt = "{:>16s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}\n"
        s += fmt.format("PlusStress:  xx", "yy", "zz", "yz", "xz", "xy")

        s += f"{vs[0][0]:>16.5f}"
        s += f"{vs[1][1]:>12.5f}"
        s += f"{vs[2][2]:>12.5f}"
        s += f"{vs[1][2]:>12.5f}"
        s += f"{vs[0][2]:>12.5f}"
        s += f"{vs[0][1]:>12.5f}\n"

        #
        s += " Feature   EFS_by	     VASP\n"
        s += f" Feature   mindist   {min_dist:.6f}\n"
        s += "END_CFG\n"

        return s

    @staticmethod
    def _get_min_dist(coords: list[Vector3D]) -> float:
        """
        Get the minimum distance between atoms in a configuration.

        Args:
            coords: atomic coordinates.

        Returns: minimum distance.
        """
        coords = np.asarray(coords)
        dists = distance_matrix(coords, coords)
        pairs_indices = np.triu_indices(dists.shape[0], k=1)
        pair_distances = dists[pairs_indices]
        min_dist = float(np.min(pair_distances))

        return min_dist


class DeepmdCollectionAdaptor(BaseDataCollectionAdaptor):
    def write(
        self,
        data: DataCollection,
        path: PathLike,
        *,
        set_size: int = None,
        seed: int = 35,
        reference_energy: dict[str, float] = None,
    ) -> list[PathLike]:
        """
        Write the data points to deepmd npy format.

        Args:
            data: data points to write.
            path: path to the directory to hold the files.
            set_size: for each system, we will split the data points into multiple
                sets with size of set_size. Per deepmd definition, a system are the
                configurations with the same number of atoms and the same species order,
                but their cells can be different. If `None`, all the data points will be
                written to one set.
            seed: random seed to shuffle the data before splitting into multiple sets.
            reference_energy: A dictionary of reference energies for each species.
                In general, one would prefer to reference energy against the free atom
                energies. If `None`, the reference energy is set to zero.
        """
        species_to_int = data.get_species_mapping()
        systems = self._group_to_systems(data)

        path = to_path(path)

        for i, stm in enumerate(systems):
            if set_size is not None:
                sets = self._split_to_sets(stm, set_size, seed=seed)
            else:
                sets = [stm]

            stm_dir = create_directory(path / f"system-{i:010d}")

            # write type map
            with open(stm_dir / "type_map.raw", "w") as f:
                for s in species_to_int.keys():
                    f.write(f"{s}\n")

            # write type.raw
            species_type = [species_to_int[s] for s in stm[0].configuration.species]
            with open(stm_dir / "type.raw", "w") as f:
                for t in species_type:
                    f.write(f"{t}\n")

            # write set
            for j, current_set in enumerate(sets):
                set_dir = create_directory(stm_dir / f"set.{j:03d}")

                box = []
                coord = []
                energy = []
                force = []
                virial = []
                for dp in current_set:
                    v = stress_to_virial(
                        dp.property.stress, dp.configuration.cell, sign=-1
                    )

                    box.append(np.ravel(dp.configuration.cell))
                    coord.append(np.ravel(dp.configuration.coords))
                    energy.append(dp.get_cohesive_energy(reference_energy))
                    force.append(np.ravel(dp.property.forces))
                    virial.append(np.ravel(v))

                np.save(set_dir / "box.npy", np.asarray(box))
                np.save(set_dir / "coord.npy", np.asarray(coord))
                np.save(set_dir / "energy.npy", np.asarray(energy))
                np.save(set_dir / "force.npy", np.asarray(force))
                np.save(set_dir / "virial.npy", np.asarray(virial))

        return [path]

    @staticmethod
    def _group_to_systems(data: DataCollection) -> list[list[DataPoint]]:
        """
        Group the data points into systems.

        Per deepmd definition, a system are the configurations with the same number of
        atoms and the same species order, but their cells can be different.

        Returns:
            A list of systems, each system is a list of data points with the same
            species order.
        """

        def species_string(d: DataPoint):
            return "-".join(d.configuration.species)

        sorted_data_points = sorted(data.data_points, key=species_string)
        groups = [list(g) for _, g in groupby(sorted_data_points, key=species_string)]

        return groups

    @staticmethod
    def _split_to_sets(
        datapoints: list[DataPoint], set_size: int, seed: int = 35
    ) -> list[list[DataPoint]]:
        """
        Split the data points into multiple sets with size of set_size.

        Args:
            datapoints: the data points to split.
            set_size: the size of each set.
            seed: random seed to shuffle the data before splitting.

        Returns:
            A list of sets, each set is a list of data points.
        """
        random.seed(seed)
        random.shuffle(datapoints)

        sets = []
        for i in range(0, len(datapoints), set_size):
            sets.append(datapoints[i : i + set_size])

        return sets


class DeepMDCollectionAdaptorOld(BaseDataCollectionAdaptor):
    def write(
        self,
        data: list[DataPoint],
        path: PathLike,
        *,
        reference_energy: dict[str, float] = None,
    ) -> list[PathLike]:
        """
        Write the data points to raw files, a set of raw files for each data point.

        Args:
            data: data points to write.
            path: path to a directory to hold the files.
            reference_energy: A dictionary of reference energies for each species.
                In general, one would prefer to reference energy against the free atom
                energies. If `None`, the reference energy is set to zero.
        """

        directory = create_directory(path)

        dset_species: list[list[str]] = [dv.configuration.species for dv in data]
        sorted_dset_species = sorted(dset_species, key=str)

        # True/False of whether previous element in sorted list is the same as the
        # current one
        res = [
            sub1 == sub2
            for sub1, sub2 in zip(sorted_dset_species, sorted_dset_species[1:])
        ]

        species_set = sorted(set(itertools.chain.from_iterable(dset_species)))
        enum_dict = {v: i for i, v in enumerate(species_set)}

        filenames: list[PathLike] = []
        counter = 0
        conf = [dp.configuration for dp in data]
        prop = [dp.property for dp in data]
        for index, ent in enumerate(sorted_dset_species):
            if index == 0:  # makes first directory
                f = directory.joinpath(f"datapoint-{index:010d}")
                os.mkdir(f)
                write_deepmd_raws(
                    f,
                    data[index],
                    conf[index],
                    prop[index],
                    ent,
                    enum_dict,
                    reference_energy,
                )
                filenames.append(f)

            elif res[index - 1]:  # goes into previous directory and adds new lines
                counter = counter + 1
                f = directory.joinpath(f"datapoint-{index-counter:010d}")
                write_deepmd_raws(
                    f,
                    data[index],
                    conf[index],
                    prop[index],
                    ent,
                    enum_dict,
                    reference_energy,
                )
            else:
                counter = 0
                f = directory.joinpath(f"datapoint-{index:010d}")
                os.mkdir(f)
                write_deepmd_raws(
                    f,
                    data[index],
                    conf[index],
                    prop[index],
                    ent,
                    enum_dict,
                    reference_energy,
                )
                filenames.append(f)

        return filenames


# TODO, add fitting data as input such that it becomes a connector?
# TODO, this should be moved to somewhere else, a specific directory for
class DataCollectionWriteMaker(Maker):
    """Maker to wrap the write function of data collection as a jobflow job.

    Args:
        adaptor: the data collection adaptor to use to write the data points.
        data: the data points to write.
        path: path to a file to write the data points to or a directory to hold the
            files. The exact behavior depends on the adaptor.
        reference_energy: A dictionary of reference energies for each species.
            In general, one would prefer to reference energy against the free atom
            energies. If `None`, the reference energy is set to zero.
    """

    adaptor: BaseDataCollectionAdaptor
    data: list[DataPoint]
    path: PathLike
    reference_energy: dict[str, float] = None
    extra_kwargs: dict[str, Any] = None

    @job
    def make(self):
        """
        Write the data points to a file.

        Returns: a list of path(s) to the file(s).
        """

        if self.extra_kwargs is None:
            kwargs = {}
        else:
            kwargs = self.extra_kwargs

        return self.adaptor.write(
            self.data,
            self.path,
            reference_energy=self.reference_energy,
            **kwargs,
        )


def write_deepmd_raws(
    path: PathLike,
    data: DataPoint,
    data_conf: Configuration,
    data_prop: Property,
    data_species: list[str],
    enum_dict,
    reference_energy: dict[str, float] = None,
):
    """
    Write configuration info to multiple files in raw file_format.

    Args:
        path: path name of the raw files directory
        data_conf: datapoint configuration
        data_prop: datapoint properties
        data_species: species of atoms
        enum_dict: enumeration dictionary for type_map and type
        reference_energy: A dictionary of reference energies for each species.
    """

    def array_to_string(data):
        d = np.asarray(data)
        d = d.flatten()
        s = ""
        for i in d:
            s += str(i) + " "
        return s

    filenames = [
        os.path.join(path, "energy.raw"),
        os.path.join(path, "force.raw"),
        os.path.join(path, "coord.raw"),
        os.path.join(path, "virial.raw"),
        os.path.join(path, "box.raw"),
    ]
    np.savetxt(os.path.join(path, "type_map.raw"), data_species, fmt="%s")

    np.savetxt(
        os.path.join(path, "type.raw"), [enum_dict[n] for n in data_species], fmt="%d"
    )
    with open(filenames[0], "a") as fe, open(filenames[1], "a") as ff, open(
        filenames[2], "a"
    ) as fc, open(filenames[3], "a") as fv, open(filenames[4], "a") as fb:
        fe.write(
            array_to_string(data.get_cohesive_energy(reference_energy=reference_energy))
            + "\n"
        )
        ff.write(array_to_string(data_prop.forces) + "\n")
        fv.write(array_to_string(data_prop.stress) + "\n")
        fc.write(array_to_string(data_conf.coords) + "\n")
        fb.write(array_to_string(data_conf.cell) + "\n")


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


def stress_to_virial(stress: Matrix3D, cell: Matrix3D, sign: float = 1.0) -> np.ndarray:
    """
    Convert stress tensor to virial tensor.

    Args:
        stress: Stress tensor.
        cell: Unit cell.
        sign: Sign of the virial tensor. Default to 1. Use -1 to get virial to VASP
            convention.

    Returns:
        Virial tensor.
    """
    volume = np.abs(np.dot(cell[0], np.cross(cell[1], cell[2])))

    virial = sign * np.asarray(stress) * volume

    return virial
