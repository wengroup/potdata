"""Example to sample structures from an MD trajectory."""

import numpy as np
from pymatgen.core import Structure

from potdata.samplers import DBSCANStructureSampler, SliceSampler
from potdata.transformations import M3gnetMDTransformation


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

    # apply a DBSCAN sampler to further sample the structures
    dbscan_sampler = DBSCANStructureSampler(
        species_to_select=["Mg"],
        pca_dim=2,
        dbscan_kwargs={"eps": 20, "min_samples": 10},
        core_ratio="auto",
        reachable_ratio=0.2,
        noisy_ratio=1.0,
    )
    structures = dbscan_sampler.sample(structures)

    dbscan_sampler.plot(show=True)


if __name__ == "__main__":
    sample_md_trajectory()
