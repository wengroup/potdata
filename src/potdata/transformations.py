"""
This module implements transformation operations on structures to generate new
structures.

The transformations are similar to pymatgen.transformations.standard_transformations.
"""
import abc
from pathlib import Path
from typing import Optional
import numpy as np
from monty.dev import requires
from pymatgen.analysis.elasticity import Strain
from pymatgen.core.structure import Structure
from pymatgen.core.tensors import symmetry_reduce
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.standard_transformations import (
    DeformStructureTransformation,
)
from pymatgen.transformations.transformation_abc import AbstractTransformation
import re
try:
    import m3gnet
except ImportError:
    m3gnet = None
try:
    import pyace
except ImportError:
    pyace = None

__all__ = [
    "StrainTransformation",
    "PerturbTransformation",
    "BaseMDTransformation",
    "M3gnetMDTransformation"
    "ACEMDTransformation",
]


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
            if strain_states is not None
            else self._get_default_strain_states()
        )
        self.strain_magnitudes = (
            np.asarray(strain_magnitudes)
            if strain_magnitudes is not None
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

        # select symmetry reduced strains
        if self.sym_reduce:
            strain_mapping = symmetry_reduce(strains, structure, symprec=self.symprec)
            strains = list(strain_mapping.keys())
        deformations = [s.get_deformation_matrix() for s in strains]

        # strain the structure
        deformed_structures = []
        for d in deformations:
            dst = DeformStructureTransformation(deformation=d)
            deformed_structures.append(
                {"structure": dst.apply_transformation(structure), "deformation": d}
            )

        return deformed_structures

    @property
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
            here the other key value pair is just the index of the newly generated
            structure.
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

    @property
    def is_one_to_many(self) -> bool:
        return True

    def inverse(self):
        return None

    def _perturb_a_structure(self, structure: Structure):
        s = structure.copy()
        s.perturb(distance=self.high, min_distance=self.low)
        return s


class BaseMDTransformation(AbstractTransformation):
    """
    Perform a molecular dynamics simulation to get structures along the trajectory.

    Args:
        ensemble: ensemble for MD. Options are "nvt" and "npt".
        temperature: temperature for MD, in K.
        timestep: timestep for MD, in fs.
        steps: number of MD steps.
    """

    def __init__(
        self,
        ensemble: str = "nvt",
        temperature: float = 300,
        timestep: float = 1,
        steps: int = 1000,
        potential_filename: Path | str = 'output_potential.yaml',
        potential_asi_filename: Path | str = 'output_potential.asi',
        gamma_range: Optional[tuple[float, float]] = None,
    ):
        self.ensemble = ensemble
        self.temperature = temperature
        self.timestep = timestep
        self.steps = steps
        self.potential_filename = potential_filename
        self.potential_asi_filename = potential_asi_filename
        self.gamma_range = gamma_range

    def apply_transformation(self, structure: Structure) -> list[dict]:
        """
        Returns:
            A list of structures from the trajectory.
        """
        structures = self.run_md(structure)
        indices = list(range(len(structures)))

        return [{"structure": s, "index": i} for s, i in zip(structures, indices)]

    @abc.abstractmethod
    def run_md(self, structure: Structure) -> list[Structure]:
        """
        Run a molecular dynamics simulation on a structure.

        Args:
            structure: Structure to run simulation on.

        Returns:
            Trajectory object.
        """

    @property
    def is_one_to_many(self) -> bool:
        return True

    def inverse(self):
        return None


class M3gnetMDTransformation(BaseMDTransformation):
    """Molecular dynamics transformation using m3gnet."""

    @requires(
        m3gnet,
        "`m3gnet` is needed for this transformation. To install it, see "
        "https://github.com/materialsvirtuallab/m3gnet",
    )
    def run_md(
        self,
        structure: Structure,
        trajectory_filename: str = "md.traj",
        log_filename: str = "md.log",
        **kwargs,
    ) -> list[Structure]:
        from ase.io import Trajectory as Trajectory
        from m3gnet.models import MolecularDynamics
        from pymatgen.io.ase import AseAtomsAdaptor

        supported = ["nvt", "npt"]
        if self.ensemble.lower() not in supported:
            raise ValueError(
                f"Unknown ensemble: {self.ensemble}. Supported are {supported}."
            )

        md = MolecularDynamics(
            atoms=structure,
            ensemble=self.ensemble,
            temperature=self.temperature,
            timestep=self.timestep,
            trajectory=trajectory_filename,
            logfile=log_filename,
            **kwargs,
        )

        md.run(steps=self.steps)
        ase_traj = Trajectory(trajectory_filename)

        structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in ase_traj]

        return structures

class ACEMDTransformation(BaseMDTransformation):
    """Molecular dynamics transformation using ACE.

    Args:
        taut (float): time constant for Berendsen temperature coupling
        loginterval (int): write to log file every interval steps
        append_trajectory (bool): Whether to append to prev trajectory
        gamma_value (γ): known as extrapolation grade, is often used to         
        indicate a model's predictive power and how well the model fits
        the training data. If γ is smaller than γselect, it implies no
        active learning actions. If γ is between γselect and γbreak, it
        indicates reliability for training set extension, and if γ is
        larger than γbreak, it is risky and trigger termination of the
        simulation. Usually, γselect equal to 2 and γbreak equal to 10,
        which is the first and second float in gamma_range
        max_gamma_value: look at the maximum gamma of each configuration,
        we want to include an entire configuration for labeling, not
        individual atoms. So, as long as the maximum gamma of a
        configuration is large enough, we should include this configuration

    """

    @requires(
        pyace,
        "`ACE` is needed for this transformation. To install it, see "
        "https://pacemaker.readthedocs.io/en/latest/",
    )
    def run_md(
        self,
        structure: Structure,
        trajectory_filename: str = "md.traj",
        log_filename: str = "md.log",
        timestep: float = 1.0,
        taut: Optional[float] = None,
        loginterval: int = 1,
        append_trajectory: bool = False,
        gamma_values_filename: str = 'gamma_values.txt',
    ) -> list[Structure]:
        from ase.io import Trajectory as Trajectory
        from ase.md.nvtberendsen import NVTBerendsen
        from ase import units
        from pymatgen.io.ase import AseAtomsAdaptor
        from pyace import PyACECalculator

        atoms = AseAtomsAdaptor.get_atoms(structure)
        # Initialize ACE calculator
        calc = PyACECalculator(self.potential_filename)
        calc.set_active_set(self.potential_asi_filename)
        atoms.set_calculator(calc)
        self.calc = calc
        
        taut = 100 * timestep * units.fs

        self.dyn = NVTBerendsen(
            atoms,
            timestep=timestep * units.fs,
            temperature_K=self.temperature,
            taut=taut,
            trajectory=trajectory_filename,
            logfile=log_filename,
            loginterval=loginterval,
            append_trajectory=append_trajectory,
        )
        self.dyn.run(self.steps)
        
        # Convert ASE trajectory to pymatgen structures
        ase_traj = Trajectory(trajectory_filename)
        structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in ase_traj]

        # Save structures and gamma values to a file
        with open(gamma_values_filename, 'w') as f:
            for i, structure in enumerate(structures, start=0):
                atoms = AseAtomsAdaptor.get_atoms(structure)
                atoms.set_calculator(calc)
                energy = atoms.get_potential_energy()
                gamma = calc.results["gamma"]
                f.write(f'Step {i}:\n Gamma value:\n {gamma},\n Energy: {energy}\n')

        # Calculate and save max gamma values per step
        with open(gamma_values_filename, 'r') as file:
            content = file.read()

        matches = re.findall(r'Step (\d+):[^[]+Gamma value:[^[]+\[([^\]]+)]', content, re.DOTALL)

        max_gamma_values_output_filename = 'max_gamma_values_per_step.txt'
        with open(max_gamma_values_output_filename, 'w') as output_file:
            for match in matches:
                step = int(match[0])
                gamma_values = [float(val) for val in match[1].split()]
                max_gamma_value = max(gamma_values)
                output_file.write(f"Step {step}:\n Gamma value:\n[{max_gamma_value}]\n")

        # Calculate and save max gamma values between γselet and γbreak
        if self.gamma_range:
            with open(max_gamma_values_output_filename, 'r') as file:
                content = file.read()

            matches = re.findall(r'Step (\d+):[^[]+Gamma value:[^[]+\[([^\]]+)]', content, re.DOTALL)

            max_gamma_between_γselet_and_γbreak_output_filename = 'max_gamma_between_γselet_and_γbreak_steps.txt'
            between_count = 0
            with open(max_gamma_between_γselet_and_γbreak_output_filename, 'w') as output_file:
                for match in matches:
                    step = int(match[0])
                    max_gamma_value = float(match[1])
                    if self.gamma_range[0] <= max_gamma_value <= self.gamma_range[1]:
                        output_file.write(f"Step {step}: Max Gamma value between γselet and γbreak: {max_gamma_value}\n")
                        between_count += 1
                    else:
                        output_file.write(f"Step {step}: Max Gamma value not between γselet and γbreak.\n")

            print(f"Results saved to {max_gamma_values_output_filename}")
            print(f"Results saved to {max_gamma_between_γselet_and_γbreak_output_filename}")
            print(f"Total number of steps with Max Gamma values between γselet and γbreak: {between_count}")

        return structures

    def __reduce__(self):
        # Exclude unpickleable objects during pickling
        state = self.__dict__.copy()
        
        # Add more attributes to exclude
        exclude_attributes = ['calc', 'dyn', '...']  # Add other relevant attributes

        for attr in exclude_attributes:
            state.pop(attr, None)

        return (self.__class__, (), state)

    def calculate_between_count(self) -> int:
        """Calculate the between_count based on the saved gamma values file."""
        # Assuming gamma values are saved in a file named 'gamma_values.txt'
        with open('gamma_values.txt', 'r') as file:
            content = file.read()

        matches = re.findall(r'Step (\d+):[^[]+Gamma value:[^[]+\[([^\]]+)]', content, re.DOTALL)

        between_count = 0
        for match in matches:
            max_gamma_value = max(map(float, match[1].split()))
            if 2 <= max_gamma_value <= 10:
                between_count += 1

        return between_count

# TODO this is obsolete, need to be adapted
# @dataclass
# class CoherentStructureMaker(StructureComposeMaker):
#     """
#     Maker to generate multiple strained structures.
#
#     Args:
#         interface_gap: Distance between two slabs when interfaced (default: 2 angstroms)
#         slab1_thickness: Thickness of structure 1 slab; scaled by how many unit cells.
#             Note this value corresponds to a multiplier.e.g slab1_thickness: 2 implies
#             two unit cell of slab1 is used in the interfaced structure. (default: 1)
#         slab2_thickness: Same properties as slab1, except for second slab structure
#         slab1_miller_indices_upper_limit: highest miller indices value
#             e.g slab1_miller_indices_upper_limit:3 yields only miller indices matches
#             below (333) (default: 2)
#         slab2_miller_indices_upper_limit: Same properties as slab1, except for second
#             slab structure
#     """
#
#     name: str = "coherent interface structure job"
#     interface_gap: float = 2
#     slab1_thickness: float = 1
#     slab2_thickness: float = 1
#     slab1_miller_max: int = 2
#     slab2_miller_max: int = 2
#     miller_indices_matches: list[tuple] = None
#
#     def miller_matches(self, struc1, struc2) -> set[tuple[Any, Any]]:
#         sa = SubstrateAnalyzer(self.slab1_miller_max, self.slab2_miller_max)
#         matches = list(sa.calculate(substrate=struc1, film=struc2))
#         new_match = []
#         for match in matches:
#             new_match.append((match.film_miller, match.substrate_miller))
#         return set(new_match)
#
#     def compose_structure(  # type: ignore[override]
#         self, structure: tuple[Structure, Structure]
#     ) -> list[Structure]:
#         """
#         Generates list of all interfaced structures from Structure1 and Structure2.
#
#         Args:
#             structure: Parent structures to create the interface.
#
#         Returns:
#             Interfaced structures generated from the two parent structure.
#         """
#         struct1 = SpacegroupAnalyzer(structure[0]).get_conventional_standard_structure()
#         struct2 = SpacegroupAnalyzer(structure[1]).get_conventional_standard_structure()
#
#         miller_matches = self.miller_matches(struct1, struct2)
#         all_interfaces = []
#         for millerindex in miller_matches:
#             cib = CoherentInterfaceBuilder(
#                 substrate_structure=struct1,
#                 film_structure=struct2,
#                 film_miller=millerindex[1],
#                 substrate_miller=millerindex[0],
#             )
#             for termination in cib.terminations:
#                 interfaces = list(
#                     cib.get_interfaces(
#                         termination,
#                         gap=self.interface_gap,
#                         vacuum_over_film=self.interface_gap,
#                     )
#                 )
#                 all_interfaces.extend(interfaces)
#
#         # The following lines removes structures that are too similar made from CoherentInterfaceBuilder
#         shortened_interfaces = []
#         for interface in all_interfaces:
#             if interface not in shortened_interfaces:
#                 shortened_interfaces.append(interface)
#
#         return shortened_interfaces
