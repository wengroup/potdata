"""Define fitting data point that contains configuration, property, weight etc."""


from typing import Union

import numpy as np
from ase import Atoms
from ase.io import Trajectory
from emmet.core.tasks import OutputDoc
from emmet.core.vasp.calculation import IonicStep
from pydantic import BaseModel, Field
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from potdata import __version__
from potdata._typing import Matrix3D, Vector3D
from potdata.samplers import BaseSampler
from potdata.utils.enum import ValueEnum
from potdata.utils.units import kbar_to_eV_per_A_cube

__all__ = ["Property", "Weight", "DataPoint", "DataCollection"]


class TaskType(ValueEnum):
    """Vasp calculation task types."""

    Static = "Static"
    Structure_Optimization = "Structure Optimization"
    Molecule_Dynamics = "Molecular Dynamics"
    Unrecognized = "Unrecognized"


# TODO, need to come up with a better way of tracking the provenance of data
class Provenance(BaseModel):
    """Provenance of the data point."""

    job_uuid: str = Field(
        None,
        description="The uuid of the job that generated the data.",
    )
    task_type: Union[TaskType, str] = Field(
        None, description="Task type of the job that generated the data."
    )
    frame: Union[None, int] = Field(
        None,
        description="From a relaxation or molecular dynamics trajectory, multiple "
        "configurations can be extracted. This field can be used to give the frame of "
        "the trajectory that the data point corresponds to.",
    )


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

    structure: Structure = Field(description="An atomic configuration.")

    property: Property = Field(
        description="Properties associated with the configuration."
    )

    weight: Weight = Field(None, description="Weight for the configuration.")

    label: str = Field(None, description="A description of the data data point.")

    provenance: Provenance = Field(None, description="Provenance of the data point.")

    # TODO DataPoints is stored in the DB using a JobStore, then uuid should be assigned
    # Anyways, UUID should be handled by Provenance
    # directly by the jobflow. See https://github.com/materialsproject/jobflow/blob/fb522a24cb695dc4cc20c72ae7e1ac77fc5ea7cf/src/jobflow/core/job.py#L601
    # If we do not use JobStore, then we can use suuid to generate uuid.
    # uuid: str = Field(default_factory=suuid, description="A uuid for the data point.")

    _schema: str = Field(
        __version__,
        description="Version of potdata used to create the document.",
        alias="schema",
    )

    @classmethod
    def from_output_summary(
        cls,
        output: OutputDoc,
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
            structure=output.structure,
            property=prop,
            weight=weight,
            label=label,
            provenance=Provenance(job_uuid=job_uuid, task_type=task_type, frame=None),
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
            structure=ionic_step.structure,
            property=prop,
            weight=weight,
            label=label,
            provenance=Provenance(job_uuid=job_uuid, task_type=task_type, frame=frame),
        )

    @classmethod
    def from_ase_atoms(
        cls,
        atoms: Atoms,
        weight: Weight = None,
        job_uuid: str | None = None,
        task_type: str | None = None,
        frame: int = None,
        label: str = None,
    ):
        """Get a data point from an ASE atoms object."""
        structure = AseAtomsAdaptor.get_structure(atoms)

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces().tolist()
        stress = atoms.get_stress()
        stress = [
            [stress[0], stress[5], stress[4]],
            [stress[5], stress[1], stress[3]],
            [stress[4], stress[3], stress[2]],
        ]

        prop = Property(energy=energy, forces=forces, stress=stress)

        return cls(
            structure=structure,
            property=prop,
            weight=weight,
            label=label,
            provenance=Provenance(job_uuid=job_uuid, task_type=task_type, frame=frame),
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
            for s, count in self.structure.composition.items():
                coh_e -= count * reference_energy[s.symbol]

        return coh_e


class DataCollection(BaseModel):
    """A data collection that contains a set of DataPoint."""

    data_points: list[DataPoint] = Field(
        description="A sequence of data points that constitutes the data collection. "
    )

    # uuid: str = Field(
    #     default_factory=suuid, description="A uuid for the data collection."
    # )

    label: str = Field(None, description="A description of the data collection.")

    reference_energy: dict[str, float] = Field(
        None,
        description="Reference energy for each species. Typically, one would use the "
        "free atom energies as the reference energies.",
    )

    @classmethod
    def from_ase_trajectory(
        cls, trajectory: Trajectory, sampler: BaseSampler = None, **kwargs
    ):
        """
        Get a data collection from an ASE trajectory.

        Args:
            trajectory: An ASE trajectory.
            sampler: A sampler that can be used to sample the trajectory.
            kwargs: Additional keyword arguments to be passed to the DataCollection.
        """
        if "data_points" in kwargs:
            raise ValueError(
                "`data_points` cannot be specified when creating from trajectory. It "
                "will be automatically generated."
            )

        indices = list(range(len(trajectory)))
        if sampler is not None:
            indices = sampler.sample(indices)

        data_points = [
            DataPoint.from_ase_atoms(trajectory[i], frame=i) for i in indices
        ]

        return cls(data_points=data_points, **kwargs)

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, item):
        return self.data_points[item]

    def __iter__(self):
        return iter(self.data_points)

    def __add__(self, other):
        data_points = self.data_points + other.data_points
        label = f"Combined from: `{self.label}` and `{other.label}`"

        if other.reference_energy is not None:
            if self.reference_energy is not None:
                for k, v in other.reference_energy.items():
                    if k in self.reference_energy and v != self.reference_energy[k]:
                        raise ValueError(
                            f"Reference energy for species `{k}` is different, get "
                            f"{v} and {self.reference_energy[k]}."
                        )
                ref_e = self.reference_energy.copy().update(other.reference_energy)
            else:
                ref_e = other.reference_energy
        else:
            ref_e = self.reference_energy

        return DataCollection(
            data_points=data_points, label=label, reference_energy=ref_e
        )

    def get_species(self) -> list[str]:
        """Get a list of species in the data collection."""
        species_set = set()
        for dp in self.data_points:
            species_set.update(dp.structure.symbol_set)

        return sorted(species_set)

    def get_species_mapping(self) -> dict[str, int]:
        """Get a mapping of species string to integer type."""
        species = self.get_species()
        mapping = {s: i for i, s in enumerate(species)}

        return mapping
