import numpy as np
from pymatgen.core import Lattice, Structure


def get_coords_range(
    coords: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """
    Get the range of the coords of all atoms in a structure.

    Args:
        coords: coords array of shape (n, 3), where n is the number of atoms

    Returns:
        x_range: tuple of min and max x coords
        y_range: tuple of min and max y coords
        z_range: tuple of min and max z coords
    """
    x_range = (coords[:, 0].min(), coords[:, 0].max())
    y_range = (coords[:, 1].min(), coords[:, 1].max())
    z_range = (coords[:, 2].min(), coords[:, 2].max())

    return x_range, y_range, z_range


def create_dummy_cell(coords: np.ndarray, padding: float = 100) -> np.ndarray:
    """
    Create a dummy cell such that a new structure using this dummy cell will not have
    any periodic images of the original structure.

    For example, this can be used to represent a molecule without any periodic
    interactions using a crystal structure that is inherently periodic.

    Padding is added to range of the coords of the original structure to create the
    dummy cell. The default of 100 might be too unnecessarily large for most cases.
    If you know the cutoff distance r_cut of the interactions you are interested in,
    you can use a value slightly larger than that, e.g. 1.1 * r_cut.

    Args:
        coords: coords array of shape (n, 3), where n is the number of atoms
        padding: padding to add to the original structure.

    Returns:
        3x3 numpy array representing the dummy cell.
        [[x_max - x_min + padding, 0, 0]
         [0, y_max - y_min + padding, 0]
         [0, 0, z_max - z_min + padding]]
        where x/y/z_max and x/y/z_min are the max and min x/y/z coords of the original
        structure.
    """
    x_range, y_range, z_range = get_coords_range(coords)

    cell = np.array(
        [
            [x_range[1] - x_range[0] + padding, 0, 0],
            [0, y_range[1] - y_range[0] + padding, 0],
            [0, 0, z_range[1] - z_range[0] + padding],
        ]
    )

    return cell


def create_lattice(
    cell: np.ndarray | None, pbc: tuple[bool, bool, bool] | None, coords: np.ndarray
) -> tuple[Lattice, bool]:
    """
    Create a pymatgen Lattice object from either the cell or the coords array.

    If cell is not None, then the Lattice object is created using the cell and pbc.
    Otherwise, a dummy cell is created using the coords array.

    Args:
        cell: 3x3 numpy array representing the cell.
        pbc: tuple of 3 booleans representing the periodic boundary conditions.
        coords: coords array of shape (n, 3), where n is the number of atoms

    Returns:
        lattice: pymatgen Lattice object
        has_cell: boolean indicating whether the Lattice object was created using the
            cell or the coords array.

    """
    if cell is not None:
        if pbc is not None:
            lattice = Lattice(cell, pbc)
        else:
            lattice = Lattice(cell)

        has_cell = True
    else:
        lattice = create_dummy_cell(coords)
        has_cell = False

    return lattice, has_cell


def get_cell_and_pbc(
    structure: Structure,
) -> tuple[np.ndarray | None, tuple[bool, bool, bool] | None]:
    """
    Get the cell and pbc from a pymatgen Structure object.

    This depends on whether Structure.property['use_lattice'] is True or False.

    Args:
        structure: pymatgen Structure object

    Returns:
        cell: 3x3 numpy array representing the cell.
        pbc: tuple of 3 booleans representing the periodic boundary conditions.
    """

    if structure.property.get("use_lattice", True):
        cell = structure.lattice.matrix
        pbc = structure.lattice.pbc
    else:
        cell = None
        pbc = None

    return cell, pbc
