"""VASP data adaptor."""

import warnings

import pandas as pd
from pymatgen.core import Structure

from potdata._typing import PathLike
from potdata.schema.datapoint import DataCollection, DataPoint, Property
from potdata.utils.path import to_path

from .base import BaseDataCollectionAdaptor
from .utils import create_lattice, get_cell_and_pbc


class JSONCollectionAdaptor(BaseDataCollectionAdaptor):
    """
    A data collection adaptor that reads/writes data points from/to JSON file.

    The JSON file is storted like using a pandas DataFrame, where a single key is used
    for all data points. e.g.
    {'coords': [coords of structure 1, coords of structure 2, ...],
        'species': [species of structure 1, species of structure 2 ...],
        ...
     'energy': [energy of structure 1, energy of structure 2, ...],
        ...
    }

    Args:
        coords_key: key for the coordinates in the json file. Default to "coords".
        species_key: key for the species in the json file. Default to "species".
        cell_key: key for the cell in the json file. Default to "cell". If None, no
            cell will be read/write.
        pbc_key: key for the periodic boundary conditions in the json file. Default to
            "pbc". If None, no pbc will be read/write.
        energy_key: key for the energy in the json file. Default to "energy".
        forces_key: key for the forces in the json file. Default to "forces". If None,
            or the key cannot be found, forces will not be read/write.
        stress_key: key for the stress in the json file. Default to "stress". If None,
            or the key cannot be found, stress will not be read/write.
    """

    def __init__(
        self,
        coords_key: str = "coords",
        species_key: str = "species",
        cell_key: str = "cell",
        pbc_key: str = "pbc",
        energy_key: str = "energy",
        forces_key: str = "forces",
        stress_key: str = "stress",
    ):
        # structure keys
        self.coords_key = coords_key
        self.cell_key = cell_key
        self.species_key = species_key
        self.pbc_key = pbc_key

        # property keys
        self.energy_key = energy_key
        self.forces_key = forces_key
        self.stress_key = stress_key

    def read(self, path: PathLike) -> DataCollection:  # type: ignore[override]
        """
        Read the data collection from a JSON file.

        It can be a list of DataPoints or a DataCollection.

        Args:
            path: path to the JSON file.

        Returns:
            A list of data points.
        """
        df = pd.read_json(to_path(path))

        for k in [self.coords_key, self.species_key, self.energy_key]:
            if k not in df.columns:
                raise ValueError(f"Cannot find key `{k}` in the file `{path}`.")

        for k in [self.cell_key, self.pbc_key, self.forces_key, self.stress_key]:
            if k is not None and k not in df.columns:
                warnings.warn(f"Cannot find key `{k}` in the file `{path}`. Ignored.")

        data_points = []
        for i, row in df.iterrows():
            # structure
            coords = row[self.coords_key]
            species = row[self.species_key]
            cell = row.get(self.cell_key, None)
            pbc = row.get(self.pbc_key, None)

            # TODO, the below line should be done for all adaptors
            lattice, use_lattice = create_lattice(cell, pbc, coords)

            structure = Structure(
                lattice=lattice,
                species=species,
                coords=coords,
                coords_are_cartesian=True,
                properties={"use_lattice": use_lattice},
            )

            # property
            y = {}
            y["energy"] = row[self.energy_key]
            y["forces"] = row.get(self.forces_key, None)
            y["stress"] = row.get(self.stress_key, None)

            data_points.append(DataPoint(structure=structure, property=Property(**y)))

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
        Write the data collection to a json file.


        """
        data_points = data.data_points

        data_dict = {
            self.coords_key: [dp.structure.cart_coords for dp in data_points],
            self.species_key: [dp.structure.species for dp in data_points],
            self.energy_key: [
                dp.get_cohesive_energy(reference_energy=reference_energy)
                for dp in data_points
            ],
        }

        if self.cell_key is not None:
            cell = []
            pbc = []
            for dp in data_points:
                c, p = get_cell_and_pbc(dp.structure)
                cell.append(c)
                pbc.append(p)
            data_dict[self.cell_key] = cell
            data_dict[self.pbc_key] = pbc

        if self.forces_key is not None:
            data_dict[self.forces_key] = [dp.property.forces for dp in data_points]
        if self.stress_key is not None:
            data_dict[self.stress_key] = [dp.property.stress for dp in data_points]

        df = pd.DataFrame(data_dict)

        df.to_json(path)

        return path
