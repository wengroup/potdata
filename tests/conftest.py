import os
import shutil
import tempfile
from pathlib import Path

import pytest
from atomate2.vasp.schemas.task import TaskDocument
from monty.serialization import loadfn
from potdata.schema.datapoint import DataCollection, DataPoint
from pymatgen.core import Structure


@pytest.fixture(scope="session")
def test_data_dir():
    return Path(__file__).resolve().parent / "test_data"


@pytest.fixture(scope="session")
def debug_mode():
    return False


@pytest.fixture(scope="session")
def clean_dir(debug_mode):
    """
    Create a new temp directory.

    When passed as an argument of a test function, the test will automatically be run
    in this directory.
    """
    old_cwd = Path.cwd()
    working_dir = tempfile.mkdtemp()
    os.chdir(working_dir)
    try:
        yield
    finally:
        if debug_mode:
            print(f"Tests ran in {working_dir}")
        else:
            os.chdir(old_cwd)
            shutil.rmtree(working_dir)


@pytest.fixture(scope="function")
def tmp_dir(debug_mode):
    """Same as clean_dir() but is fresh for every test."""
    old_cwd = Path.cwd()
    working_dir = tempfile.mkdtemp()
    os.chdir(working_dir)
    try:
        yield
    finally:
        if debug_mode:
            print(f"Tests ran in {working_dir}")
        else:
            os.chdir(old_cwd)
            shutil.rmtree(working_dir)


@pytest.fixture(scope="session")
def relax_task_doc(test_data_dir) -> TaskDocument:
    filename = test_data_dir / "schema" / "relax_task_doc.json.gz"
    doc = loadfn(filename)

    return doc


@pytest.fixture(scope="session")
def relax_fitting_data_points(relax_task_doc) -> list[DataPoint]:
    datapoints = [
        DataPoint.from_ionic_step(s)
        for i, s in enumerate(relax_task_doc.calcs_reversed[0].output.ionic_steps)
    ]

    return datapoints


@pytest.fixture(scope="session")
def relax_fitting_data_collection(relax_fitting_data_points) -> DataCollection:
    return DataCollection(data_points=relax_fitting_data_points)


@pytest.fixture(scope="session")
def Si_structure():
    struct = Structure(
        lattice=[
            [3.348898, 0.0, 1.933487],
            [1.116299, 3.157372, 1.933487],
            [0.0, 0.0, 3.866975],
        ],
        species=["Si", "Si"],
        coords=[[0.25, 0.25, 0.25], [0, 0, 0]],
    )

    return struct
