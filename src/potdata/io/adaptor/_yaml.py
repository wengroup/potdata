"""VASP data adaptor."""

import copy

from monty.json import jsanitize
from monty.serialization import dumpfn, loadfn

from potdata._typing import PathLike
from potdata.schema.datapoint import DataCollection, DataPoint
from potdata.utils.dataops import remove_none_from_dict
from potdata.utils.path import to_path

from .base import BaseDataCollectionAdaptor


class YAMLCollectionAdaptor(BaseDataCollectionAdaptor):
    def read(self, path: PathLike) -> DataCollection:  # type: ignore[override]
        """
        Read the data collection from a YAML file.

        It can be a list of DataPoints or a DataCollection.

        Args:
            path: path to the YAML file.

        Returns:
            A list of data points.
        """
        path = to_path(path)
        data = loadfn(path, fmt="yaml")

        if isinstance(data, (list, tuple)):
            datapoints = [DataPoint(**d) for d in data]
            dc = DataCollection(data_points=datapoints)
        else:
            dp = data.pop("data_points")
            datapoints = [DataPoint(**d) for d in dp]
            dc = DataCollection(data_points=datapoints, **data)

        return dc

    # TODO add dealing with reference_energy
    def write(
        self,
        data: DataCollection,
        path: PathLike,
        *,
        reference_energy: dict[str, float] = None,
        as_list: bool = False,
    ) -> list[PathLike]:
        """
        Write the data points to a YAML file.

        Args:
            data: data points to write.
            path: path to the YAML file.
            reference_energy: A dictionary of reference energies for each species.
                In general, one would prefer to reference energy against the free atom
                energies. If `None`, the reference energy is set to zero.
            as_list: Whether to write the data as a list data points or as a data
                collection (which contains other metadata such as the label).
        """
        path = to_path(path)

        if as_list:
            out = [remove_none_from_dict(dp.dict()) for dp in data.data_points]
        else:
            out = copy.copy(data)
            out = remove_none_from_dict(out.dict())  # type: ignore

        out = jsanitize(out, strict=True)
        dumpfn(out, path, fmt="yaml")

        return [path]
