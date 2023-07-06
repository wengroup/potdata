import numpy as np
import pytest

from potdata.schema.datapoint import DataCollection, DataPoint, Property, Weight


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


def test_data_point(Si_structure, Si_property, Si_weight):
    prop = Property(**Si_property)
    weight = Weight(**Si_weight)

    dp = DataPoint(structure=Si_structure, property=prop, weight=weight)

    prop = dp.property
    assert prop.energy == Si_property["energy"]
    assert np.allclose(prop.forces, Si_property["forces"])
    assert np.allclose(prop.stress, Si_property["stress"])

    weight = dp.weight
    assert weight.energy_weight == Si_weight["energy_weight"]
    assert weight.config_weight == Si_weight["config_weight"]
    assert np.allclose(weight.forces_weight, Si_weight["forces_weight"])
    assert np.allclose(weight.stress_weight, Si_weight["stress_weight"])

    assert dp.get_cohesive_energy(reference_energy={"Si": 0.1}) == 0.8


def test_data_collection(Si_structure, Si_property, Si_weight):
    prop = Property(**Si_property)
    weight = Weight(**Si_weight)

    data = []
    for _ in range(2):
        dp = DataPoint(structure=Si_structure, property=prop, weight=weight)
        data.append(dp)

    # data collection as a list of actual data points
    dc = DataCollection(data_points=data)
    assert len(dc) == len(data)
