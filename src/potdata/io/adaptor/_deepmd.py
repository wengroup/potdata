"""VASP data adaptor."""

import random
from itertools import groupby

import numpy as np

from potdata._typing import PathLike
from potdata.schema.datapoint import DataCollection, DataPoint
from potdata.utils.path import create_directory, to_path

from .base import BaseDataCollectionAdaptor
from .utils import stress_to_virial


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
            species_type = [species_to_int[str(s)] for s in stm[0].structure.species]
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
                        dp.property.stress, dp.structure.lattice.matrix, sign=-1
                    )

                    box.append(np.ravel(dp.structure.lattice.matrix))
                    coord.append(np.ravel(dp.structure.cart_coords))
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
            return "-".join([s.symbol for s in d.structure.species])

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
