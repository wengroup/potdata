import os
import shutil
import tempfile
from pathlib import Path

import pytest
from monty.serialization import loadfn
from pymatgen.core import Structure

from potdata.schema.datapoint import DataCollection


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
def fitting_data_collection(test_data_dir) -> DataCollection:
    test_data_file = test_data_dir / "io" / "data_collection.json.gz"
    dc = loadfn(test_data_file)
    return dc


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
