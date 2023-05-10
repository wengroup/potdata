"""Define fitting data point that contains configuration, property, weight etc."""

from typing import Union

import numpy as np
from ase import Atoms
from atomate2.vasp.schemas.calc_types.enums import TaskType
from atomate2.vasp.schemas.calculation import IonicStep
from atomate2.vasp.schemas.task import OutputSummary
from pydantic import BaseModel, Field
from pymatgen.core.structure import Structure

from potdata import __version__
from potdata._typing import Matrix3D, Vector3D
from potdata.utils.suuid import suuid
from potdata.utils.units import kbar_to_eV_per_A_cube


__all__ = ["Configuration", "Property", "Weight", "DataPoint", "DataCollection"]


class Provenance(BaseModel):
    """Provenance of the data point."""

    job_uuid: str = Field(
        None,
        description="The uuid of the job that generated the data.",
    )

    task_type: Union[TaskType, str] = Field(
        None,
        description="atomate2 task type of the job that generated the data, "
        "e.g. `Static`, `Structure Optimization`, and `MD`",
    )


class Configuration(BaseModel):
    """An atomic configuration."""

    species: list[str] = Field(
        description="Species of the atoms, typically their atomic symbols, e.g. "
        "['Si', 'Si']",
    )

    coords: list[Vector3D] = Field(
        description="Cartesian coordinates of the atoms. Shape (N, 3), where N is the "
        "number of atoms in the configuration. Example Units: A."
    )

    cell: Matrix3D = Field(
        None,
        description="Cell vectors a_1, a_2, and a_3 of the simulation box. If `None`, "
        "this is a cluster without cell (typical for a molecule). Example units: A.",
    )

    pbc: Union[tuple[bool, bool, bool], tuple[int, int, int]] = Field(
        (True, True, True),
        description="Periodic boundary conditions along the three cell vectors a_1, "
        "a_2, and a_3.",
    )

    @classmethod
    def from_pymatgen_structure(cls, structure: Structure):
        return cls(
            species=[str(s) for s in structure.species],
            coords=structure.cart_coords.tolist(),
            cell=structure.lattice.matrix.tolist(),
            pbc=(True, True, True),
        )

    @classmethod
    def from_colabfit(cls):
        pass

    @classmethod
    def from_ase(cls, atoms: Atoms):
        return cls(
            species=[str(s) for s in atoms.get_chemical_symbols()],
            coords=atoms.get_positions().tolist(),
            cell=atoms.get_cell().tolist(),
            pbc=atoms.get_pbc().tolist(),
        )

    @property
    def composition_dict(self) -> dict[str, int]:
        """Get the composition of the configuration as a dictionary."""
        return {s: self.species.count(s) for s in set(self.species)}


class Property(BaseModel):
    """Properties associated with an atomic configuration."""

    energy: float = Field(
        None,
        description="Total energy of a configuration. Example units: eV.",
    )

    forces: list[Vector3D] = Field(
        None,
        description="Forces on atoms. Shape: (N, 3), where N is the number of atoms "
        "in the configuration. Example units: eV/A.",
    )

    stress: Matrix3D = Field(
        None,
        description="Stress on the simulation box. Example units: eV/A^3.",
    )


class Weight(BaseModel):
    """Weight used in loss function for a configuration."""

    energy_weight: float = Field(1.0, description="Energy weight.")
    forces_weight: Union[float, list[Vector3D]] = Field(
        1.0,
        description="Forces weight. If a float, it will be applied to each component "
        "of the forces. Otherwise, it should have the same shape as forces.",
    )
    stress_weight: Union[float, Matrix3D] = Field(
        1.0,
        description="Stress weight. If a float, it will be applied to each component "
        "of the stress. Otherwise it should have the same shape as stress.",
    )
    config_weight: float = Field(
        1.0,
        description="Configuration weight. A scaling factor to be multiplied with "
        "energy/forces/stress weight.",
    )


class DataPoint(BaseModel):
    """A data point of a pair of configuration and the property associated with it."""

    configuration: Configuration = Field(description="An atomic configuration.")

    property: Property = Field(
        description="Properties associated with the configuration."
    )

    weight: Weight = Field(None, description="Weight for the configuration.")

    provenance: Provenance = Field(None, description="Provenance of the data point.")

    frame: Union[None, int] = Field(
        None,
        description="From a relaxation or molecular dynamics trajectory, multiple "
        "configurations can be extracted. This field gives the frame of the trajectory "
        "that the data point corresponds to.",
    )

    label: str = Field(None, description="A description of the data data point.")

    uuid: str = Field(default_factory=suuid, description="A uuid for the data point.")

    _schema: str = Field(
        __version__,
        description="Version of potdata used to create the document.",
        alias="schema",
    )

    @classmethod
    def from_output_summary(
        cls,
        output: OutputSummary,
        weight: Weight = None,
        job_uuid: str | None = None,
        task_type: str | None = None,
        label: str = None,
    ):
        """
        Get data point from an atomate2 OutputSummary.

        For relaxation and molecular dynamics job, this corresponds to the last ionic
        step.
        """
        config = Configuration.from_pymatgen_structure(output.structure)

        # units conversion from kbar to eV/A^3
        # VASP uses compression as the positive direction for stress, opposite to the
        # convention. Therefore, the sign is flipped with the minus sign.
        ratio = -kbar_to_eV_per_A_cube()

        # energy, forces, and stress are in eV, eV/A, and eV/A^3
        prop = Property(
            energy=output.energy,
            forces=output.forces,
            stress=(ratio * np.asarray(output.stress)).tolist(),
        )

        return cls(
            configuration=config,
            property=prop,
            weight=weight,
            provenance=Provenance(job_uuid=job_uuid, task_type=task_type),
            frame=None,
            label=label,
        )

    @classmethod
    def from_ionic_step(
        cls,
        ionic_step: IonicStep,
        weight: Weight = None,
        job_uuid: str | None = None,
        task_type: str | None = None,
        frame: int = None,
        label: str = None,
    ):
        """Get a data point from :obj:`atomate2.vasp.schemas.calculation.IonicStep`."""
        config = Configuration.from_pymatgen_structure(ionic_step.structure)

        # units conversion from kbar to eV/A^3
        # VASP uses compression as the positive direction for stress, opposite to the
        # convention. Therefore, the sign is flipped with the minus sign.
        ratio = -kbar_to_eV_per_A_cube()

        # energy, forces, and stress are in eV, eV/A, and eV/A^3
        prop = Property(
            energy=ionic_step.e_0_energy,
            forces=ionic_step.forces,
            stress=(ratio * np.asarray(ionic_step.stress)).tolist(),
        )

        return cls(
            configuration=config,
            property=prop,
            weight=weight,
            provenance=Provenance(job_uuid=job_uuid, task_type=task_type),
            frame=frame,
            label=label,
        )

    def get_cohesive_energy(self, reference_energy: dict[str, float] = None) -> float:
        """Get the cohesive energy of the configuration.

        Args:
            reference_energy: A dictionary of reference energies for each species.
                In general, one would prefer to reference energy against the free atom
                energies.
        """
        if reference_energy is None:
            coh_e = self.property.energy
        else:
            coh_e = self.property.energy
            for s, count in self.configuration.composition_dict.items():
                coh_e -= count * reference_energy[s]

        return coh_e


class DataCollection(BaseModel):
    """A data collection that contains a set of DataPoint."""

    data_points: list[DataPoint] = Field(
        description="A sequence of data points that constitutes the data collection. "
    )

    uuid: str = Field(
        default_factory=suuid, description="A uuid for the data collection."
    )

    label: str = Field(None, description="A description of the data collection.")

    reference_energy: dict[str, float] = Field(
        None,
        description="Reference energy for each species. Typically, one would use the "
        "free atom energies as the reference energies.",
    )

    def __len__(self):
        return len(self.data_points)

    def get_species(self) -> list[str]:
        """Get a list of species in the data collection."""
        species_set = set()
        for dp in self.data_points:
            species_set.update(dp.configuration.species)

        return sorted(species_set)

    def get_species_mapping(self) -> dict[str, int]:
        """Get a mapping of species string to integer type."""
        species = self.get_species()
        mapping = {s: i for i, s in enumerate(species)}

        return mapping
