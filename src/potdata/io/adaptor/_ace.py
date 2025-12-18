"""VASP data adaptor."""

import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor

from potdata._typing import PathLike
from potdata.schema.datapoint import DataCollection, DataPoint, Property
from potdata.utils.path import to_path

from .base import BaseDataCollectionAdaptor


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
                structure=AseAtomsAdaptor.get_structure(row["ase_atoms"]),
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
                    AseAtomsAdaptor.get_atoms(dp.structure) for dp in datapoints
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
