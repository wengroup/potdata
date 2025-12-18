"""VASP data adaptor."""

import warnings

import pandas as pd
from pymatgen.core import Structure

from potdata._typing import PathLike
from potdata.schema.datapoint import DataCollection, DataPoint, Property
from potdata.utils.path import to_path

from .base import BaseDataCollectionAdaptor


class PymatgenCollectionAdaptor(BaseDataCollectionAdaptor):
    """
    A data collection adaptor that reads/writes data points from/to pymatgen structure
    and the corresponding targets.

    The input/output will be a json file.

    Args:
        structure_key: key for the structure in the json file. Default to "structure".
        energy_key: key for the energy in the json file. Default to "energy".
        forces_key: key for the forces in the json file. Default to "forces". If None,
            or the key cannot be found, forces will not be read/write.
        stress_key: key for the stress in the json file. Default to "stress". If None,
            or the key cannot be found, stress will not be read/write.
    """

    def __init__(
        self,
        structure_key: str = "structure",
        energy_key: str = "energy",
        forces_key: str = "forces",
        stress_key: str = "stress",
    ):
        self.structure_key = structure_key
        self.energy_key = energy_key
        self.forces_key = forces_key
        self.stress_key = stress_key

    def read(self, path: PathLike) -> DataCollection:  # type: ignore[override]
        """
        Read the data collection.

        Args: Path to read the data collection. This can be a path to a file or path
        to a directory.

        """
        df = pd.read_json(to_path(path))
        df["structure"] = df[self.structure_key].apply(lambda s: Structure.from_dict(s))

        # check required keys are present
        for k in [self.structure_key, self.energy_key]:
            if k not in df.columns:
                raise ValueError(f"Cannot find key `{k}` in the file `{path}`.")

        # check if optional keys are present
        for k in [self.forces_key, self.stress_key]:
            if k is not None and k not in df.columns:
                warnings.warn(f"Cannot find key `{k}` in the file `{path}`. Ignore it.")

        data_points = []
        for i, row in df.iterrows():
            y = {}
            y["energy"] = row[self.energy_key]
            y["forces"] = row.get(self.forces_key, None)
            y["stress"] = row.get(self.stress_key, None)

            dp = DataPoint(structure=row["structure"], property=Property(**y))
            data_points.append(dp)

        dc = DataCollection(data_points=data_points)

        return dc

    def write(
        self,
        data: DataCollection,
        path: PathLike,
        *,
        reference_energy: dict[str, float] = None,
    ) -> PathLike:
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

        data_points = data.data_points

        data_dict = {self.structure_key: [dp.structure.as_dict() for dp in data_points]}

        energy = [
            dp.get_cohesive_energy(reference_energy=reference_energy)
            for dp in data_points
        ]
        forces = [dp.property.forces for dp in data_points]
        stress = [dp.property.stress for dp in data_points]

        if None not in energy:
            data_dict[self.energy_key] = energy
        if None not in forces:
            data_dict[self.forces_key] = forces
        if None not in stress:
            data_dict[self.stress_key] = stress

        df = pd.DataFrame(data_dict)

        path = to_path(path)
        if path.is_dir():
            path = path.joinpath("data.json")
        df.to_json(path)

        return path
