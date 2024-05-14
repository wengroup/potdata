"""Example to parse fitting data from VASP and write to other formats, such at MTP,
DeePMD, YAML, and ACE.
"""

from pathlib import Path

from potdata._typing import PathLike
from potdata.io.adaptor import (
    DeepmdCollectionAdaptor,
    ExtxyzCollectionAdaptor,
    MTPCollectionAdaptor,
    VasprunCollectionAdaptor,
    YAMLCollectionAdaptor,
)


def vasprun_to_mtp(
    input_path: PathLike,
    output_path: PathLike,
    reference_energy: dict[str, float] = None,
):
    """Convert vasprun.xml files to MTP cfg files.

    Args:
        input_path: Path to a directory where vasprun.xml files are located in itself
            and its subdirectories.
        output_path: Path to the output MTP cfg file.
        reference_energy: A dictionary of reference energies for each species.
            In general, one would prefer to reference energy against the free atom
            energies. If `None`, the reference energy is set to zero. E.g. {"Si":
            -8.0, 'O': -5.0}.

    """
    data_collection = VasprunCollectionAdaptor().read(
        input_path, index=None, name_pattern="vasprun.xml"
    )

    writer = MTPCollectionAdaptor()
    writer.write(data_collection, output_path, reference_energy=reference_energy)


def vasprun_to_yaml(
    input_path: PathLike,
    output_path: PathLike,
    reference_energy: dict[str, float] = None,
):
    """Convert vasprun.xml files to YAML files.

    Args:
        input_path: Path to a directory where vasprun.xml files are located in itself
            and its subdirectories.
        output_path: Path to the output YAML file.
        reference_energy: A dictionary of reference energies for each species.
            In general, one would prefer to reference energy against the free atom
            energies. If `None`, the reference energy is set to zero. E.g. {"Si":
            -8.0, 'O': -5.0}.

    """
    datapoints = VasprunCollectionAdaptor().read(
        input_path, index=None, name_pattern="vasprun.xml"
    )
    adaptor = YAMLCollectionAdaptor()
    adaptor.write(datapoints, output_path, reference_energy=reference_energy)
    # datapoinsts = adaptor.read(output_path)


def vasprun_to_deepmd(
    input_path: PathLike,
    output_path: PathLike,
    reference_energy: dict[str, float] = None,
    set_size: int = None,
    seed: int = 35,
):
    """Convert vasprun.xml files to Deepmd dataset.

    Args:
        input_path: Path to a directory where vasprun.xml files are located in itself
            and its subdirectories.
        output_path: Path to the output Deepmd dataset.
        reference_energy: A dictionary of reference energies for each species.
            In general, one would prefer to reference energy against the free atom
            energies. If `None`, the reference energy is set to zero. E.g. {"Si":
            -8.0, 'O': -5.0}.
        set_size: Number of data points in each set. If `None`, all data points are
            put into one set.
        seed: Random seed for splitting data into sets. Ignored if `set_size=None`.
    """
    datapoints = VasprunCollectionAdaptor().read(
        input_path, index=None, name_pattern="vasprun.xml"
    )

    adaptor = DeepmdCollectionAdaptor()
    adaptor.write(
        datapoints,
        output_path,
        reference_energy=reference_energy,
        set_size=set_size,
        seed=seed,
    )


def vasprun_to_xyz(
    input_path: PathLike,
    output_path: PathLike,
    reference_energy: dict[str, float] = None,
    stress_format: str = "full",
):
    """Convert vasprun.xml files to XYZ files.

    Args:
        input_path: Path to a directory where vasprun.xml files are located in itself
            and its subdirectories.
        output_path: Path to the output XYZ file.
        reference_energy: A dictionary of reference energies for each species.
            In general, one would prefer to reference energy against the free atom
            energies. If `None`, the reference energy is set to zero. E.g. {"Si":
            -8.0, 'O': -5.0}.

    """
    datapoints = VasprunCollectionAdaptor().read(
        input_path, index=None, name_pattern="vasprun.xml"
    )
    adaptor = ExtxyzCollectionAdaptor()
    adaptor.write(
        datapoints,
        output_path,
        reference_energy=reference_energy,
        stress_format=stress_format,
    )


if __name__ == "__main__":
    path = Path.cwd() / "../../tests/test_data/vasp/Si_double_relax"

    # vasprun_to_mtp(input_path=path, output_path="~/Desktop/mtp.cfg")
    # vasprun_to_yaml(input_path=path, output_path="~/Desktop/dataset.yaml")
    # vasprun_to_deepmd(input_path=path, output_path="~/Desktop/deepmd_data")

    vasprun_to_xyz(input_path=path, output_path="~/Desktop/xyz", stress_format="voigt")
