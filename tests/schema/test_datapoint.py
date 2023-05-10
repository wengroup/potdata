import numpy as np
import pytest

from potdata.schema.datapoint import (
    Configuration,
    DataCollection,
    DataPoint,
    Property,
    Weight,
)


@pytest.fixture
def Si_property():
    return {
        "energy": 1.0,
        "forces": [[0.0, 0.0, 1.0], [0, 0, 2.0]],
        "stress": np.ones((3, 3)).tolist(),
    }


@pytest.fixture
def Si_weight():
    return {
        "energy_weight": 1.0,
        "forces_weight": np.ones((2, 3)).tolist(),
        "stress_weight": np.ones((3, 3)).tolist(),
        "config_weight": 1.0,
    }


def test_data_pint(Si_structure, Si_property, Si_weight):
    conf = Configuration.from_pymatgen_structure(Si_structure)
    prop = Property(**Si_property)
    weight = Weight(**Si_weight)

    dp = DataPoint(configuration=conf, property=prop, weight=weight)

    ref_cell = [
        [3.348898, 0.0, 1.933487],
        [1.116299, 3.157372, 1.933487],
        [0.0, 0.0, 3.866975],
    ]
    ref_frac_coords = [[0.25, 0.25, 0.25], [0, 0, 0]]
    ref_cart_coords = np.dot(ref_frac_coords, ref_cell)

    conf = dp.configuration
    assert np.allclose(conf.cell, ref_cell)
    assert conf.pbc == (True, True, True)
    assert conf.species == ["Si", "Si"]
    assert np.allclose(conf.coords, ref_cart_coords)

    prop = dp.property
    assert prop.energy == Si_property["energy"]
    assert np.allclose(prop.forces, Si_property["forces"])
    assert np.allclose(prop.stress, Si_property["stress"])

    weight = dp.weight
    assert weight.energy_weight == Si_weight["energy_weight"]
    assert weight.config_weight == Si_weight["config_weight"]
    assert np.allclose(weight.forces_weight, Si_weight["forces_weight"])
    assert np.allclose(weight.stress_weight, Si_weight["stress_weight"])

    # check uuid is automatically created
    assert isinstance(dp.uuid, str)

    assert dp.get_cohesive_energy(reference_energy={"Si": 0.1}) == 0.8


def test_data_collection(Si_structure, Si_property, Si_weight):
    conf = Configuration.from_pymatgen_structure(Si_structure)
    prop = Property(**Si_property)
    weight = Weight(**Si_weight)

    data = []
    for _ in range(2):
        dp = DataPoint(configuration=conf, property=prop, weight=weight)
        data.append(dp)

    # data collection as a list of actual data points
    dc1 = DataCollection(data_points=data)
    assert isinstance(dc1.uuid, str)
