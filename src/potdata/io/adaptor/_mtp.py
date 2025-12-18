"""VASP data adaptor."""

import warnings

import numpy as np
from scipy.spatial import distance_matrix

from potdata._typing import PathLike, Vector3D
from potdata.schema.datapoint import DataCollection, DataPoint
from potdata.utils.path import to_path

from .base import BaseDataCollectionAdaptor
from .utils import create_dummy_cell, get_cell_and_pbc, stress_to_virial


class MTPCollectionAdaptor(BaseDataCollectionAdaptor):
    def write(
        self,
        data: DataCollection,
        path: PathLike,
        *,
        reference_energy: dict[str, float] = None,
        cell_padding: float = None,
    ) -> PathLike:
        """
        Write the data points to MTP cfg format.

        Args:
            data: data points to write.
            path: path to the file written.
            reference_energy: A dictionary of reference energies for each species.
                In general, one would prefer to reference energy against the free atom
                energies. If `None`, the reference energy is set to zero.
            cell_padding: padding to add to the cell. MTP can only work with periodic
                structures with a cell. This is a workaround to create a cell for
                structures without a cell (e.g. molecular structures). The cell is
                created by adding `cell_padding` to the maximum distance between atoms
                in each of the x, y, z directions.
                Note that for structures with a cell, the cell is used as is and this
                argument is ignored.
        Returns: path to the file written.
        """

        species_map = data.get_species_mapping()

        s = ""
        for dp in data.data_points:
            s += self.as_string(dp, species_map, reference_energy, cell_padding) + "\n"

        path = to_path(path)
        with open(path, "w") as f:
            f.write(s)

        return path

    def as_string(
        self,
        dp: DataPoint,
        species_map: dict[str, int],
        reference_energy: dict[str, float] = None,
        cell_padding: float = None,
    ) -> str:
        """
        Convert a data point to a string in MTP cfg format.

        Args:
            dp: data point to convert.
            species_map: A dictionary of species string to species integer.
            reference_energy: A dictionary of reference energies for each species.
            cell_padding: padding to add to the cell.

        Returns:
            A configuration in MTP cfg format as a string.
        """
        coords = dp.structure.cart_coords
        size = len(coords)
        energy = dp.get_cohesive_energy(reference_energy=reference_energy)
        min_dist = self._get_min_dist(coords)

        s = "BEGIN_CFG\n"
        s += " Size\n"
        s += f"{size:>5}\n"

        s += " Supercell\n"
        cell, _ = get_cell_and_pbc(dp.structure)

        # MTP requires a cell. So if the structure does not have a cell, we create a
        # cell with very large lattice paramters
        if cell is None:
            if cell_padding is None:
                raise RuntimeError(
                    "For structures without a cell (e.g. molecuels), `cell_padding` "
                    "must be provided to create a dummy cell since MTP can only work "
                    "with periodic structures. The cell is created by adding "
                    "`cell_padding` to the maximum distance between atoms in each of "
                    "the x, y, z directions.\n"
                    "The padding should be chosen such at the atoms do not interact "
                    "with their periodic images. As a result, the padding should be "
                    "larger than the cutoff radius of your model. For exmaple, "
                    "1.1*r_cut would be a good choice.\n"
                    "Note also, if you increase your model cutoff radius, you probably "
                    "need to increase the padding accordingly."
                )
            cell = create_dummy_cell(coords, cell_padding)
            warnings.warn(
                f"No cell information found. Create a dummy supercell {cell}.",
                stacklevel=2,
            )

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
            zip(dp.structure.species, coords, dp.property.forces)
        ):
            fmt = (
                "{:>19d}{:>14d}{:>14.6f}{:>14.6f}{:>14.6f}{:>14.6f}{:>14.6f}{:>14.6f}\n"
            )
            s += fmt.format(i + 1, species_map[sp.symbol], *co, *fo)

        # energy
        s += " Energy\n"
        s += f"     {energy:.12f}\n"

        # stress
        if dp.property.stress is not None:
            vs = stress_to_virial(
                dp.property.stress, dp.structure.lattice.matrix, sign=-1
            )
            fmt = "{:>16s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}\n"
            s += fmt.format("PlusStress:  xx", "yy", "zz", "yz", "xz", "xy")

            s += f"{vs[0][0]:>16.5f}"
            s += f"{vs[1][1]:>12.5f}"
            s += f"{vs[2][2]:>12.5f}"
            s += f"{vs[1][2]:>12.5f}"
            s += f"{vs[0][2]:>12.5f}"
            s += f"{vs[0][1]:>12.5f}\n"

            s += " Feature   EFS_by	     VASP\n"
        else:
            s += " Feature   EF_by	     VASP\n"

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
