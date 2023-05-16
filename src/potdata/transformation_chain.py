"""
This module implements chained transformation operations on structures to generate
new structures.

A transformation chain contains a sequence of transformations to be applied
sequentially. The transformations in a chain can be unit transformation or another
transformation chain.
"""

from pymatgen.core import Structure
from pymatgen.transformations.transformation_abc import AbstractTransformation


class TransformationChain(AbstractTransformation):
    """
    Chain a sequence of transformations and treat them as a single transformation.

    The transformations are applied sequentially. The input structure is fed to the
    first transformation, and its output structure(s) is/are fed as the input for the
    second transformation, and so on.

    Args:
        transformations: List of transformations to chain.
    """

    def __init__(self, transformations: list[AbstractTransformation]):
        self.transformations = transformations

    def apply_transformation(self, structure: Structure) -> list[dict]:
        """
        Apply the transformation to a structure.

        Args:
            structure: Structure to transform.

        Returns:
            A list of transformed structures. Each transformed structure is specified
            in a dict:

            {"structure": structure, "chain_steps": [step1, step2, ...]}

            where chain_steps stores the history of the transformation. Each step is a
            dict corresponding to a transformation applied in the chain:

            {"transformation": transformation,
             "input_structure": in_structure,
             "output_structure": out_structure,
             "tranformation_meta": other_data_for_a_specific_structure
            }

            where `transformation` specifies the transformation class applied in this
            step. `input_structure` and `output_structure` are the structure before and
            after the transformation. Note that a transforamtion class may generate
            multiple structures when `transformation.is_one_to_many` is True. In this
            case, although `transformation` and `input_structure` is the same,
            `output_structure` will be different for each structure generated.
            In additional, `transformation_meta` specifies the metadata for a specific
            structrue obtained using a one-to-many transformation. For non one-to-many
            transformation, `transformation_meta` is an empty dict.
        """

        # steps keep track of the history of the transformation
        transformed_structures = [{"structure": structure, "chain_steps": []}]

        # loop over transformations
        for i, t in enumerate(self.transformations):
            new_transformed_structures = []

            # loop over current structures
            for struct_step_dict in transformed_structures:
                struct = struct_step_dict.pop("structure")
                steps = struct_step_dict.pop("chain_steps")

                structures = t.apply_transformation(struct)

                # If is_one_to_many, structures is a list of dict
                # If not is_one_to_many, structures is a Structure, and we make it a
                # list of dict here
                if not t.is_one_to_many:
                    structures = [{"structure": structures}]

                # loop over newly generated structures
                for j, new_struct_meta_dict in enumerate(structures):
                    new_struct = new_struct_meta_dict.pop("structure")
                    new_meta = new_struct_meta_dict

                    step_dict = {
                        "transformation": t.as_dict(),
                        "input_structure": struct.as_dict(),
                        "output_structure": new_struct.as_dict(),
                        "transformation_meta": new_meta,
                    }
                    new_steps = steps.copy()
                    new_steps.append(step_dict)

                    d = {"structure": new_struct, "chain_steps": new_steps}
                    new_transformed_structures.append(d)

            transformed_structures = new_transformed_structures

        return transformed_structures

    def is_one_to_many(self):
        for t in self.transformations:
            if t.is_one_to_many():
                return True
        return False

    def inverse(self):
        return None
