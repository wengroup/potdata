"""VASP data adaptor."""

import numpy as np
from pymatgen.io.vasp import Vasprun

from potdata._typing import PathLike
from potdata.schema.datapoint import DataCollection, DataPoint, Property
from potdata.utils.dataops import slice_sequence
from potdata.utils.path import to_path
from potdata.utils.units import kbar_to_eV_per_A_cube

from .base import BaseDataCollectionAdaptor, BaseDataPointAdaptor


class VasprunAdaptor(BaseDataPointAdaptor):
    """VASP vasprun.xml adaptor."""

    def read(
        self, path: PathLike, index: int | list[int] | slice | None = -1
    ) -> list[DataPoint]:
        """Read vasprun.xml file into a list of data points.

        Args:
            path: Path to the vasprun.xml file.
            index: Index of the ionic step to read. Default to select the last ionic
                step. If `None`, all ionic steps are read. See
                :obj:`potdata.utils.dataops.slice_sequence` for more information on
                advanced selection.
        """
        vasprun = Vasprun(
            path,
            parse_dos=False,
            parse_eigen=False,
            parse_projected_eigen=False,
            parse_potcar_file=False,
        )
        ionic_steps, _ = slice_sequence(vasprun.ionic_steps, slicer=index)

        # units conversion from kbar to eV/A^3
        # VASP uses compression as the positive direction for stress, opposite to the
        # convention. Therefore, the sign is flipped with the minus sign.
        ratio = -kbar_to_eV_per_A_cube()

        datapoints = []
        for step in ionic_steps:
            dp = DataPoint(
                structure=step["structure"],
                property=Property(
                    forces=step["forces"],
                    stress=(ratio * np.asarray(step["stress"])).tolist(),
                    energy=step["e_0_energy"],
                ),
            )
            datapoints.append(dp)

        return datapoints


class VasprunCollectionAdaptor(BaseDataCollectionAdaptor):
    def read(  # type: ignore[override]
        self,
        path: PathLike,
        index: int | list[int] | slice | None = -1,
        name_pattern: str = "vasprun.xml",
    ) -> DataCollection:
        """Read all vasprun.xml from a directory into a list of data points.

        Args:
            path: Path to the vasprun.xml file or directory containing vasprun.xml.
            index: Index of the ionic step to read. Default to select the last ionic
                step. See `VasprunAdaptor.read()` for more information.
            name_pattern: All files with `<name_pattern>` in the filename will be
                treated as vasprun.xml files.
        """

        adaptor = VasprunAdaptor()

        path = to_path(path)
        if path.is_file():
            filenames = [path]
        elif path.is_dir():
            filenames = [p for p in path.rglob(f"*{name_pattern}*") if p.is_file()]
        else:
            raise RuntimeError(f"Path `{path}` is not a file or directory.")

        datapoints = []
        for p in filenames:
            datapoints.extend(adaptor.read(p, index=index))

        dc = DataCollection(data_points=datapoints)

        return dc
