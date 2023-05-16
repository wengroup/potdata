import numpy as np
from potdata.transformations import PerturbTransformation, StrainTransformation


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
