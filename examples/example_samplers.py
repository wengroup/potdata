"""Example to sample structures from an MD trajectory."""

from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

from potdata.samplers import ACEGammaSampler, KMeansStructureSampler, SliceSampler
from potdata.transformations import ACEMDTransformation, M3gnetMDTransformation


def get_structure():
    """Create an example Si structure."""
    structure = Structure(
        lattice=np.array([[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]]),
        species=["Mg", "O"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )
    structure.make_supercell([3, 3, 3])  # increase the size of the structure

    return structure


def sample_md_trajectory():
    mgo_structure = get_structure()
    md_trans = M3gnetMDTransformation(temperature=300, steps=1000)
    data = md_trans.apply_transformation(mgo_structure)

    # `data` is a list of dictionaries, with each dictionary containing a structure
    structures = [d["structure"] for d in data]

    # apply a slice sampler to get a subset of the structures
    # this will sample steps 200, 202, 204, ..., 998, 1000
    slice_sampler = SliceSampler(index=slice(200, None, 2))
    structures = slice_sampler.sample(structures)

    # apply a KMeans sampler to further sample the structures
    kmeans_sampler = KMeansStructureSampler(
        kmeans_kwargs={"n_clusters": 10}, species_to_select=["Mg"], pca_dim=2, ratio=0.1
    )
    structures = kmeans_sampler.sample(structures)

    print(f"Number of structures: {len(structures)}")

    kmeans_sampler.plot(show=True)


def sample_md_trajectory_2(structure, potential, active_set):
    """Run MD using ACE and then sample with ACEGammaSampler."""
    transformation = ACEMDTransformation(
        steps=1000, potential_filename=potential, potential_asi_filename=active_set
    )
    data = transformation.apply_transformation(structure)
    structures = [d["structure"] for d in data]

    sampler = ACEGammaSampler(
        potential_filename=potential,
        potential_asi_filename=active_set,
        gamma_range=(0.5, 0.9),
        gamma_reduce="max",
        verbose=1,
    )

    structures = sampler.sample(structures)
    print(f"Number of structures: {len(structures)}")


if __name__ == "__main__":
    sample_md_trajectory()

    path = Path("~/Downloads/CONTCAR").expanduser()
    s = Poscar.from_file(path).structure
    potential = Path("~/Downloads/output_potential.yaml").expanduser().as_posix()
    active_set = Path("~/Downloads/output_potential.asi").expanduser().as_posix()
    sample_md_trajectory_2(s, potential, active_set)
