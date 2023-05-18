import numpy as np
import pytest
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer

from potdata.sampler import SliceSampler
from potdata.transformations import (
    M3gnetMDTransformation,
    PerturbTransformation,
    StrainTransformation,
)

try:
    import m3gnet
except ImportError:
    m3gnet = None


def test_strain_transformation(Si_structure):
    st = StrainTransformation(conventional=False)
    transformed_structures = st.apply_transformation(Si_structure)
    assert len(transformed_structures) == 7 * 4

    st = StrainTransformation(conventional=True)
    transformed_structures = st.apply_transformation(Si_structure)
    # only 2 for shear, because, for example, -0.1 and 0.1 are the same for shear
    assert len(transformed_structures) == 10

    for d in transformed_structures:
        assert "structure" in d
        assert "deformation" in d

    strain_states = [[1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]]
    strain_magnitudes = [0.1, 0.2]

    st = StrainTransformation(strain_states=strain_states)
    transformed_structures = st.apply_transformation(Si_structure)
    assert len(transformed_structures) == len(strain_states) * 4

    st = StrainTransformation(strain_magnitudes=strain_magnitudes)
    transformed_structures = st.apply_transformation(Si_structure)
    assert len(transformed_structures) == 7 * len(strain_magnitudes)

    st = StrainTransformation(
        strain_states=strain_states, strain_magnitudes=strain_magnitudes
    )
    transformed_structures = st.apply_transformation(Si_structure)
    assert len(transformed_structures) == len(strain_states) * len(strain_magnitudes)


def test_perturb_transformation(Si_structure):
    num_struct = 10
    low = 0.1
    high = 0.3
    pt = PerturbTransformation(num_structures=num_struct, low=low, high=high)
    transformed_structures = pt.apply_transformation(Si_structure)

    for d in transformed_structures:
        assert "structure" in d
        assert "index" in d

    # only select the atom whose initial position is not (0, 0, 0).
    # the (0, 0, 0) atom can be perturbed to the other side of the cell due to PBC.
    orig_coords = Si_structure.cart_coords[[0]]

    assert len(transformed_structures) == num_struct
    for d in transformed_structures:
        s = d["structure"]
        new_coords = s.cart_coords[[0]]

        distances = np.linalg.norm(new_coords - orig_coords, axis=1)

        assert np.max(distances) <= high
        assert np.min(distances) >= low


@pytest.mark.skipif(m3gnet is None, reason="m3gnet is not installed")
def test_md_transformation(Si_structure):
    # get a conventional cell
    sga = SpacegroupAnalyzer(Si_structure)
    structure = sga.get_conventional_standard_structure()

    # increase number of cells
    structure.make_supercell([2, 2, 2])

    mt = M3gnetMDTransformation(
        ensemble="nvt", steps=10, sampler=SliceSampler(slicer=slice(5, None, 2))
    )
    structures = mt.apply_transformation(structure)

    assert len(structures) == 3
    for s in structures:
        assert len(s) == 64
        assert s.lattice == structure.lattice
