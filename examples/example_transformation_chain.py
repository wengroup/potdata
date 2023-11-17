import numpy as np
from pymatgen.core import Structure

from potdata.transformation_chain import TransformationChain
from potdata.transformations import PerturbTransformation, StrainTransformation


def get_structure():
    """Create an example Si structure."""
    structure = Structure(
        lattice=np.array([[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]]),
        species=["Mg", "O"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )
    structure.make_supercell([3, 3, 3])  # increase the size of the structure

    return structure


def get_transformed_structures(structure):
    """
    Apply a series of transformations
        - StrainTransformation
        - PerturbTransformation
    to the structure and return the transformed structures.
    """

    st = StrainTransformation()
    pt = PerturbTransformation(num_structures=10)
    chain = TransformationChain(transformations=[st, pt])

    # `transformed` is a list of dict, with each dict containing a transformed structure
    transformed = chain.apply_transformation(structure)

    # let's see what is in the dict
    print("Keys:", transformed[0].keys())

    # transformed structures
    structures = [d["structure"] for d in transformed]

    print("Number of generated transformed structures:", len(structures))

    return structures


if __name__ == "__main__":
    structure = get_structure()
    transformed_structures = get_transformed_structures(structure)
