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

    assert len(transformed_structures) == 10 * num_perturb_struct

    ts0 = transformed_structures[0]
    assert "structure" in ts0
    assert "chain_steps" in ts0

    steps = ts0["chain_steps"]

    assert len(steps) == 2

    for step in steps:
        for name in [
            "transformation",
            "input_structure",
            "output_structure",
            "transformation_meta",
        ]:
            assert name in step

    assert "deformation" in steps[0]["transformation_meta"]
    assert "index" in steps[1]["transformation_meta"]
