"""
This module implements transformation operations on structures to generate new
structures.

The transformations are similar to pymatgen.transformations.standard_transformations.
"""

import numpy as np
from pymatgen.analysis.elasticity import Strain
from pymatgen.core.structure import Structure
from pymatgen.core.tensors import symmetry_reduce
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.standard_transformations import (
    DeformStructureTransformation,
)
from pymatgen.transformations.transformation_abc import AbstractTransformation


class StrainTransformation(AbstractTransformation):
    """
    Strain a structure in given directions by a given amount.

    Args:
        strain_states: List of Voigt-notation strain states. By default, the strains are
            along the x, y, z, yz, xz, and xy directions, i.e. the strain_states are:
              [[1, 1, 1, 0, 0, 0],    # uniform strain in all directions
               [1, 0, 0, 0, 0, 0],    # strain along 11 direction
               [0, 1, 0, 0, 0, 0],    # strain along 22 direction
               [0, 0, 1, 0, 0, 0],    # strain along 33 direction
               [0, 0, 0, 2, 0, 0],    # strain along 23 direction
               [0, 0, 0, 0, 2, 0],    # strain along 13 direction
               [0, 0, 0, 0, 0, 2]]    # strain along 12 direction
        strain_magnitudes: A list of strain magnitudes to multiply by for each strain
            state, e.g. ``[-0.02, -0.01, 0.01, 0.02]``. Alternatively, a list of
            lists can be provided, where each inner list specifies the magnitude for
            each strain state.
        conventional: Whether to transform the structure into the conventional cell.
        sym_reduce: Whether to reduce the number of deformations using symmetry.
        symprec: Symmetry precision for spglib symmetry finding.
    """

    def __init__(
        self,
        strain_states: list[list[int]] = None,
        strain_magnitudes: list[float] | list[list[float]] = None,
        conventional: bool = False,
        sym_reduce: bool = True,
        symprec: float = 0.1,
    ):
        self.strain_states = (
            np.asarray(strain_states)
            if strain_states
            else self._get_default_strain_states()
        )
        self.strain_magnitudes = (
            np.asarray(strain_magnitudes)
            if strain_states
            else self._get_default_strain_magnitudes()
        )
        self.conventional = conventional
        self.sym_reduce = sym_reduce
        self.symprec = symprec

    def apply_transformation(self, structure: Structure) -> list[dict]:
        """
        Apply the transformation to a structure.

        Args:
            structure: Structure to transform.

        Returns:
            A list of dict {"structure": structure, "other_key": other_value }, where
            there could be multiple other key value pairs. The other key value pairs
            are necessary information to reconstruct the transformation. For example,
            here for the strain transformation, the other key value pair is the
            deformation gradient matrix.
        """

        if self.conventional:
            sga = SpacegroupAnalyzer(structure, symprec=self.symprec)
            structure = sga.get_conventional_standard_structure()

        if self.strain_magnitudes.ndim == 1:
            magnitudes = [self.strain_magnitudes] * len(self.strain_states)
        else:
            magnitudes = self.strain_magnitudes

        strains = [
            Strain.from_voigt(m * s)
            for s, mag in zip(self.strain_states, magnitudes)
            for m in mag
        ]

        # remove zero strains
        strains = [s for s in strains if (np.abs(s) > 1e-10).any()]

        deformations = [s.get_deformation_matrix() for s in strains]
        if self.sym_reduce:
            deformation_mapping = symmetry_reduce(
                deformations, structure, symprec=self.symprec
            )
            deformations = list(deformation_mapping.keys())

        # strain the structure
        deformed_structures = []
        for d in deformations:
            dst = DeformStructureTransformation(deformation=d)
            deformed_structures.append(
                {"structure": dst.apply_transformation(structure), "deformation": d}
            )

        return deformed_structures

    def is_one_to_many(self) -> bool:
        return True

    def inverse(self):
        return None

    @staticmethod
    def _get_default_strain_states():
        return np.asarray(
            [
                [1, 1, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 2, 0, 0],
                [0, 0, 0, 0, 2, 0],
                [0, 0, 0, 0, 0, 2],
            ]
        )

    @staticmethod
    def _get_default_strain_magnitudes():
        return np.asarray([-0.02, -0.01, 0.01, 0.02])


class PerturbTransformation(AbstractTransformation):
    """
    Perturb the coordinates of atoms in a structure.

    The atoms are perturbed by a random distance in a random direction. The distance is
    sampled from a uniform distribution between ``low`` and ``high``.


    Args:
        low: Lower bound of the distance to perturb atoms by, in Angstrom.
        high: Upper bound of the distance to perturb atoms by, in Angstrom.
        num_structures: Number of structures to generate.
        validate_proximity: Whether to check whether the perturbed atoms are too close
            to each other. If True, the perturbation is repeated until the atoms are
            not too close to each other.
        proximity_tolerance: The minimum distance (Angstrom) between atoms allowed when
            performing the proximity check.
        trail: The number of times to try to perturb the atoms when the proximity check
            fails before giving up.
        seed: Random seed to use when generating the perturbations.
    """

    def __init__(
        self,
        low: float = 0.0,
        high: float = 0.3,
        num_structures: int = 5,
        validate_proximity: bool = True,
        proximity_tolerance: float = 0.5,
        trail: int = 10,
        seed: int = 35,
    ):
        self.low = low
        self.high = high
        self.num_structures = num_structures
        self.validate_proximity = validate_proximity
        self.proximity_tolerance = proximity_tolerance
        self.trail = trail
        self.seed = seed

        np.random.seed(self.seed)

    def apply_transformation(self, structure: Structure) -> list[dict]:
        """
        Apply the transformation to a structure.

        Args:
            structure: Structure to transform.

        Returns:
            A list of dict {"structure": structure, "other_key": other_value }, where
            there could be multiple other key value pairs. The other key value pairs
            are necessary information to reconstruct the transformation. For example,
            here the other key value pair is just the index of the newly
            generated structure.
        """
        perturbed_structures = []
        for i in range(self.num_structures):
            if self.validate_proximity:
                count = 1
                while True:
                    s = self._perturb_a_structure(structure)
                    if s.is_valid(tol=self.proximity_tolerance):
                        break
                    count += 1
                    if count > self.trail:
                        raise ValueError(
                            f"Cannot generate a valid structure after {count} trails."
                        )
            else:
                s = self._perturb_a_structure(structure)

            perturbed_structures.append({"structure": s, "index": i})

        return perturbed_structures

    def is_one_to_many(self) -> bool:
        return True

    def inverse(self):
        return None

    def _perturb_a_structure(self, structure: Structure):
        s = structure.copy()
        s.perturb(distance=self.high, min_distance=self.low)
        return s
