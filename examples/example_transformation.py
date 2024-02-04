from pathlib import Path

from pymatgen.io.vasp import Poscar

from potdata.transformations import ACEMDTransformation


def get_transformed_structures(structure, potential, active_set):
    transformation = ACEMDTransformation(
        steps=1000,
        potential_filename=potential,
        potential_asi_filename=active_set,
        gamma_range=(0.8, 1.0),
        verbose=1,
    )
    data = transformation.apply_transformation(structure)
    structures = [d["structure"] for d in data]

    return structures


if __name__ == "__main__":
    path = Path("~/Downloads/CONTCAR").expanduser()
    s = Poscar.from_file(path).structure

    potential = Path("~/Downloads/output_potential.yaml").expanduser().as_posix()
    active_set = Path("~/Downloads/output_potential.asi").expanduser().as_posix()

    structures = get_transformed_structures(s, potential, active_set)
    print("Number selected structures:", len(structures))
