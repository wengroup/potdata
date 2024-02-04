"""Generate example fitting data for test."""

import os
import tempfile
from pathlib import Path

import numpy as np
from monty.serialization import dumpfn
from pymatgen.core import Structure

from potdata.schema.datapoint import DataCollection


def get_si():
    return Structure(
        lattice=np.asarray([[0, 2.13, 2.13], [2.13, 0, 2.13], [2.13, 2.13, 0]]),
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )


def get_mgo():
    return Structure(
        lattice=np.asarray([[0, 2.13, 2.13], [2.13, 0, 2.13], [2.13, 2.13, 0]]),
        species=["Mg", "O"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
    )


def get_md_samples(structure):
    from ase.io import Trajectory
    from m3gnet.models import MolecularDynamics

    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)

        md = MolecularDynamics(
            atoms=structure,
            ensemble="nvt",
            temperature=300,
            timestep=1,
            trajectory="md.traj",
        )
        md.run(steps=5)
        traj = Trajectory("md.traj")

    return DataCollection.from_ase_trajectory(traj)


if __name__ == "__main__":
    from monty.json import jsanitize

    dc_si = get_md_samples(get_si())
    dc_mgo = get_md_samples(get_mgo())
    dc = dc_si + dc_mgo
    dc = jsanitize(dc, strict=True, enum_values=True)
    dumpfn(dc, Path("~/Desktop/data_collection.json.gz").expanduser())
