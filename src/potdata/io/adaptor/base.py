"""Base adaptor to convert data format."""

from monty.json import MSONable, jsanitize

from potdata._typing import PathLike
from potdata.schema.datapoint import DataCollection, DataPoint


class BaseDataPointAdaptor(MSONable):
    """
    Base adaptor that converts a DataPoint to other formats.

    Subclass should implement a `write()` method to convert a
    :obj:`~potdata.schema.datapoint.DataPoint` to other format, and it is optional
    for a subclass to implement a read function to convert other format to a DataPoint.
    """

    def read(self, path: PathLike) -> DataPoint | list[DataPoint]:
        """
        Read into a data point.

        Args:
            path: filename from which to read a DataPoint.
        """
        raise NotImplementedError

    def write(
        self,
        datapoint: DataPoint,
        path: PathLike,
        *,
        reference_energy: dict[str, float] = None,
        mode: str = "w",
    ) -> PathLike:
        """
        Write the data point to file.

        Args:
            datapoint: a DataPoint to convert.
            path: filename to write the DataPoint.
            reference_energy: A dictionary of reference energies for each species.
                In general, one would prefer to reference energy against the free atom
                energies. If `None`, the reference energy is set to zero.
            mode: mode to write to the file, e.g. `w` for writing and `a` for appending.

        Returns:
            Path to the file written.
        """
        raise NotImplementedError


class BaseDataCollectionAdaptor(MSONable):
    """
    Base adaptor that converts a DataCollection to other formats.

    Subclass of this class should implement a `write()` method to convert a
    :obj:`~potdata.schema.datapoint.DataCollection` to other format, and it is optional
    for a subclass to implement a read function to convert other format to a
    DataCollection.
    """

    def read(self, path: PathLike | list[PathLike]) -> DataCollection:
        """
        Read the data collection.

        Args: Path to read the data collection. This can be a path to a file or path
        to a directory.

        """
        raise NotImplementedError

    def write(
        self,
        data: DataCollection,
        path: PathLike,
        *,
        reference_energy: dict[str, float] = None,
    ) -> PathLike | list[PathLike]:
        """
        Write the data collection to file(s).

        Args:
            data: Data points to write.
            path: Path to write the data collection. This can be a path to a file or
                to a directory.
            reference_energy: A dictionary of reference energies for each species.
                In general, one would prefer to reference energy against the free atom
                energies. If `None`, the reference energy is set to zero.

        Returns:
            Path to the file or a list of filenames to which the data are written.
        """
        raise NotImplementedError
