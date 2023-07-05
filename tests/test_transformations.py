import numpy as np
import pytest

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

    # 4 volumetric strains
    # 4 uni-axial strains
    # 2 shear strains (for example, -0.1 and 0.1 are the same)
    expected = 4 + 4 + 2
    transformed_structures = st.apply_transformation(Si_structure)
    assert len(transformed_structures) == expected

    st = StrainTransformation(conventional=True)
    transformed_structures = st.apply_transformation(Si_structure)
    assert len(transformed_structures) == expected

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
    # 2 for each of v volumetric, uni-axial, and shear strains
    assert len(transformed_structures) == 2 + 2 + 2

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

    assert len(transformed_structures) == num_struct

    # only select the atom whose initial position is not (0, 0, 0).
    # the (0, 0, 0) atom can be perturbed to the other side of the cell due to PBC.
    idx = 1
    orig_coords = Si_structure.cart_coords[[idx]]

    for d in transformed_structures:
        assert "structure" in d
        assert "index" in d
        s = d["structure"]
        new_coords = s.cart_coords[[idx]]

        distances = np.linalg.norm(new_coords - orig_coords, axis=1)

        assert np.max(distances) <= high
        assert np.min(distances) >= low


@pytest.mark.skipif(m3gnet is None, reason="m3gnet is not installed")
def test_md_transformation(Si_conventional, tmpdir):
    # increase number of cells
    Si_conventional.make_supercell([2, 2, 2])

    with tmpdir.as_cwd():
        mt = M3gnetMDTransformation(
            ensemble="nvt",
            temperature=300,
            steps=5,
        )
        structures = mt.apply_transformation(Si_conventional)

    assert len(structures) == 6
    for s in structures:
        s = s["structure"]
        assert len(s) == 64
        assert s.lattice == Si_conventional.lattice
