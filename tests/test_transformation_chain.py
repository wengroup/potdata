from potdata.transformation_chain import TransformationChain
from potdata.transformations import PerturbTransformation, StrainTransformation


def test_transformation_chain(Si_structure):
    num_perturb_struct = 5
    tc = TransformationChain(
        [
            StrainTransformation(),
            PerturbTransformation(num_structures=num_perturb_struct),
        ]
    )
    transformed_structures = tc.apply_transformation(Si_structure)

    assert len(transformed_structures) == 28 * num_perturb_struct

    ts0 = transformed_structures[0]
    assert "structure" in ts0
    assert "steps" in ts0

    steps = ts0["steps"]

    assert len(steps) == 2

    assert "deformation" in steps[0]["transformation_meta"]
    assert "index" in steps[1]["transformation_meta"]
