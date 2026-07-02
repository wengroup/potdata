"""VASP data adaptor."""

import warnings

import numpy as np
from pymatgen.core import Lattice, Structure
from scipy.spatial import distance_matrix

from potdata._typing import PathLike, Vector3D
from potdata.schema.datapoint import DataCollection, DataPoint, Property
from potdata.utils.path import to_path

from .base import BaseDataCollectionAdaptor
from .utils import (
    create_dummy_cell,
    get_cell_and_pbc,
    stress_to_virial,
    virial_to_stress,
)


class MTPCollectionAdaptor(BaseDataCollectionAdaptor):
    """
    Read and write MTP config file.

    See the below link to get information its format:
    https://gitlab.com/ivannovikov/datasets_for_magnetic_MTP
    """

    def read(self, path: PathLike, type_map: dict[int, str]) -> DataCollection:
        """
        Read MTP cfg file into a DataCollection.

        This supports the read of magnetic MTP cfg files, including:
        - `magmon_x`
        - `magmon_y`
        - `magmon_z`
        - `en_der_mx`
        - `en_der_my`
        - `en_der_mz`

        Args:
            path: path to the MTP cfg file.
            type_map: A dictionary of id to chemical species string. For example, for a
                system consisting of Fe and Al with id 0 and 1 respectively, this should
                be {0: "Fe", 1: "Al"}.

        Below is a copy of the format description from the link above:

                BEGIN_CFG
         Size
            16
         Supercell
                 5.762016      0.000000      0.000000
                 0.000000      5.762016      0.000000
                 0.000000      0.000000      5.762016
         AtomData:  id type       cartes_x      cartes_y      cartes_z           fx          fy          fz      magmom_x     magmom_y     magmom_z      en_der_mx     en_der_my     en_der_mz
                     1    0       5.601141      5.634215      5.706643     0.334240    0.362515    0.091467     -0.044986     0.000000     0.000000        0.01301       0.00000       0.00000
                     2    0       2.724627      3.055367      5.753316     1.225166   -0.719191   -0.367237     -0.040295     0.000000     0.000000       -0.07494       0.00000       0.00000
                     3    0       2.891956      5.783912      2.929640     1.143133   -0.035505    0.304275     -0.037779     0.000000     0.000000       -0.08865       0.00000       0.00000
                     4    0       5.982702      2.903711      2.907225    -1.316026    0.129797    0.385899     -0.041743     0.000000     0.000000       -0.05761       0.00000       0.00000
                     5    0       1.426445      1.407661      1.435894     0.694969   -0.832026   -1.070362     -0.010516     0.000000     0.000000       -0.56632       0.00000       0.00000
                     6    1       2.916502      5.615661      5.782760     0.494663    1.097696   -0.449970      2.217293     0.000000     0.000000        0.03578       0.00000       0.00000
                     7    1       5.737528      2.839752      5.900881     0.305079    0.805712   -1.153400      2.258824     0.000000     0.000000       -0.01927       0.00000       0.00000
                     8    1       5.743693      5.881866      2.850527    -0.362800   -0.961687    0.797184      2.254575     0.000000     0.000000       -0.02846       0.00000       0.00000
                     9    1       2.887404      2.923071      2.838196    -0.179624   -0.368784    0.495499      2.279720     0.000000     0.000000       -0.01056       0.00000       0.00000
                    10    1       4.318458      1.397635      1.397865    -0.368751    0.780565    0.246046      2.082916     0.000000     0.000000       -0.05325       0.00000       0.00000
                    11    1       4.340469      4.221714      1.494782    -0.045837    0.533821    0.241160      2.033235     0.000000     0.000000        0.00071       0.00000       0.00000
                    12    1       1.479859      4.476050      1.481242    -0.812430   -0.464819   -0.062383      1.807398     0.000000     0.000000       -0.02185       0.00000       0.00000
                    13    1       1.482106      1.322786      4.433872    -0.259524    0.773825   -0.003988      2.041471     0.000000     0.000000       -0.05997       0.00000       0.00000
                    14    1       4.359945      1.623102      4.348248    -0.460788   -0.680701    0.026732      2.252012     0.000000     0.000000       -0.01946       0.00000       0.00000
                    15    1       4.369740      4.265506      4.185932    -0.193451   -0.431358    0.414076      2.086461     0.000000     0.000000       -0.01937       0.00000       0.00000
                    16    1       1.423910      4.416413      4.293970    -0.198020    0.010141    0.105003      1.880666     0.000000     0.000000        0.01034       0.00000       0.00000
         Energy
             -37537.216887746064
         PlusStress:  xx          yy          zz          yz          xz          xy
                 3.93384     5.05264     4.08282     0.10061    -0.60909    -0.45801
         Feature   mindist	2.344293
        END_CFG

        BEGIN_CFG
         Size
            16
         Supercell
                 5.778044      0.000000      0.000000
                 0.000000      5.778044      0.000000
                 0.000000      0.000000      5.761689
         AtomData:  id type       cartes_x      cartes_y      cartes_z           fx          fy          fz      magmom_x     magmom_y     magmom_z      en_der_mx     en_der_my     en_der_mz
                     1    0       5.732109      5.639314      0.052809    -0.046444    0.874276   -0.064196     -0.027727     0.000000     0.000000       -0.19045       0.00000       0.00000
                     2    0       2.968355      5.705646      0.120702     0.385984   -0.220078   -0.587812     -0.019172     0.000000     0.000000       -0.32605       0.00000       0.00000
                     3    0       2.940273      2.936691      0.090654     0.907149    1.124235   -0.439512     -0.016753     0.000000     0.000000       -0.31040       0.00000       0.00000
                     4    0       5.682996      3.008108     -0.001882     0.241985   -0.265176    0.223828     -0.028046     0.000000     0.000000       -0.19690       0.00000       0.00000
                     5    0       1.520319      1.543027      1.585386    -1.025831   -0.597896    0.267761     -0.072714     0.000000     0.000000        0.71487       0.00000       0.00000
                     6    0       1.491544      1.540716      4.322535    -0.191386   -0.639061   -1.040115     -0.011496     0.000000     0.000000       -0.26033       0.00000       0.00000
                     7    1       2.933802      5.839292      2.813952     0.340736   -0.569863    1.070871      2.241622     0.000000     0.000000        0.01222       0.00000       0.00000
                     8    1       5.780356      2.949287      2.825533     0.039138   -0.268727    1.234103      2.131255     0.000000     0.000000        0.03625       0.00000       0.00000
                     9    1       5.782667      5.762270      2.914608     0.134560    0.443599    0.236435      2.234648     0.000000     0.000000        0.03701       0.00000       0.00000
                    10    1       2.966795      2.998978      2.878713    -0.028614   -0.325023    0.673839      2.162427     0.000000     0.000000        0.03105       0.00000       0.00000
                    11    1       4.422342      1.432897      1.546149    -0.218318    0.232652   -0.441031      1.898906     0.000000     0.000000        0.00027       0.00000       0.00000
                    12    1       4.411306      4.281820      1.598984    -0.583137    0.815443   -0.389765      1.769942     0.000000     0.000000       -0.01831       0.00000       0.00000
                    13    1       1.468490      4.347112      1.565105     0.015354    0.046122   -0.506563      1.823501     0.000000     0.000000       -0.00245       0.00000       0.00000
                    14    1       4.321168      1.514021      4.338667     0.094042   -0.613829   -0.405829      2.043290     0.000000     0.000000       -0.05037       0.00000       0.00000
                    15    1       4.379411      4.370339      4.226314    -0.709090    0.057920    0.191529      1.937685     0.000000     0.000000       -0.03083       0.00000       0.00000
                    16    1       1.428621      4.380567      4.288483     0.643871   -0.094593   -0.023542      1.992550     0.000000     0.000000       -0.05301       0.00000       0.00000
         Energy
             -34208.079382760196
         PlusStress:  xx          yy          zz          yz          xz          xy
                 6.66081     7.14493     7.35465    -0.05571    -0.12654    -0.17858
         Feature   mindist	2.270323
        END_CFG
        """

        def process_one(config: str) -> DataPoint:
            """
            Process one cfg in a BEGIN_CFG and END_CFG block.
            """
            lines = config.strip().splitlines()
            size = int(lines[1].strip())
            cell_lines = lines[3:6]
            cell = []
            for line in cell_lines:
                cell.append([float(x) for x in line.strip().split()])

            # whether the cfg contains magnetic moments and its derivatives
            has_magmon = "magmom_x" in lines[6]
            has_en_der_m = "en_der_mx" in lines[6]

            atom_data_start = 7
            atom_data_end = atom_data_start + size
            coords = []
            species = []
            forces = []

            if has_magmon:
                magmon = []
            else:
                magmon = None
            if has_en_der_m:
                en_der_m = []
            else:
                en_der_m = None

            for line in lines[atom_data_start:atom_data_end]:
                val = line.strip().split()
                species_id = int(val[1])
                species.append(type_map[species_id])
                coords.append([float(val[2]), float(val[3]), float(val[4])])
                forces.append([float(val[5]), float(val[6]), float(val[7])])

                if has_magmon:
                    magmon.append([float(val[8]), float(val[9]), float(val[10])])
                if has_en_der_m:
                    en_der_m.append([float(val[11]), float(val[12]), float(val[13])])

            e_line_idx = atom_data_end + 1
            energy = float(lines[e_line_idx].strip())

            virial = None
            if "PlusStress:" in lines[e_line_idx + 1]:
                # MLIP stores VASP's extensive "FORCE on cell = -STRESS" tensor
                # in eV as PlusStress. Convert to conventional intensive stress
                # in eV/A^3 below with -PlusStress / volume.
                # See mlip-3 src/configuration.cpp and its convert_vasp_outcar
                # OUTCAR example for the "FORCE on cell =-STRESS" Total block.
                val = lines[e_line_idx + 2].strip().split()
                virial = np.zeros((3, 3), dtype=float)
                virial[0, 0] = float(val[0])
                virial[1, 1] = float(val[1])
                virial[2, 2] = float(val[2])
                virial[1, 2] = float(val[3])
                virial[0, 2] = float(val[4])
                virial[0, 1] = float(val[5])
                virial[2, 1] = virial[1, 2]
                virial[2, 0] = virial[0, 2]
                virial[1, 0] = virial[0, 1]
            if virial is not None:
                stress = virial_to_stress(virial, cell, sign=-1)
            else:
                stress = None

            # Convert to DataPoint
            structure = Structure(
                lattice=Lattice(cell, (True, True, True)),
                species=species,
                coords=coords,
                coords_are_cartesian=True,
            )

            prop = Property(
                energy=energy,
                forces=forces,
                stress=stress,
                magmon=magmon,
                magnetic_forces=en_der_m,
            )
            dp = DataPoint(structure=structure, property=prop)

            return dp

        # Read configs and split into blocks
        with open(to_path(path), "r") as f:
            data = f.read()
            configs = data.split("BEGIN_CFG")[1:]

        data_points = [process_one(c) for c in configs]
        dc = DataCollection(data_points=data_points)

        return dc

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

        has_magmon = dp.property.magmon is not None
        has_magnetic_forces = dp.property.magnetic_forces is not None

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
                    "For structures without a cell (e.g. molecules), `cell_padding` "
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
        fmt = "{:>14s}{:>5s}{:>14s}{:>14s}{:>14s}{:>14s}{:>14s}{:>14s}{:>14s}"
        names = [
            "AtomData:",
            "id",
            "type",
            "cartes_x",
            "cartes_y",
            "cartes_z",
            "fx",
            "fy",
            "fz",
        ]

        # Add magnetic moments and magnetic forces if present
        if has_magmon:
            fmt += "{:>14s}{:>14s}{:>14s}"
            names.extend(["magmom_x", "magmom_y", "magmom_z"])
        if has_magnetic_forces:
            fmt += "{:>14s}{:>14s}{:>14s}"
            names.extend(["en_der_mx", "en_der_my", "en_der_mz"])
        fmt += "\n"

        s += fmt.format(*names)

        for i in range(size):
            sp = dp.structure.species[i]
            s_id = species_map[sp.symbol]
            co = coords[i]
            fo = dp.property.forces[i]
            fmt = "{:>19d}{:>14d}{:>14.6f}{:>14.6f}{:>14.6f}{:>14.6f}{:>14.6f}{:>14.6f}"
            data = [i + 1, s_id, *co, *fo]

            # Add magnetic moments and magnetic forces if present
            if has_magmon:
                fmt += "{:>14.6f}{:>14.6f}{:>14.6f}"
                data.extend(dp.property.magmon[i])
            if has_magnetic_forces:
                fmt += "{:>14.6f}{:>14.6f}{:>14.6f}"
                data.extend(dp.property.magnetic_forces[i])
            fmt += "\n"

            s += fmt.format(*data)

        # energy
        s += " Energy\n"
        s += f"     {energy:.12f}\n"

        # stress
        if dp.property.stress is not None:
            virial = stress_to_virial(
                dp.property.stress, dp.structure.lattice.matrix, sign=-1
            )
            fmt = "{:>16s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}\n"
            s += fmt.format("PlusStress:  xx", "yy", "zz", "yz", "xz", "xy")

            s += f"{virial[0][0]:>16.5f}"
            s += f"{virial[1][1]:>12.5f}"
            s += f"{virial[2][2]:>12.5f}"
            s += f"{virial[1][2]:>12.5f}"
            s += f"{virial[0][2]:>12.5f}"
            s += f"{virial[0][1]:>12.5f}\n"

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
