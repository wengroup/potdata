"""Samplers to select a subset of objects (e.g. structures) from a sequence."""
import abc
from typing import Any, Callable, Iterable

import numpy as np
from monty.json import MSONable
from pymatgen.core.structure import Structure

from potdata.utils.dataops import serializable_slice, slice_sequence

__all__ = ["RandomSampler", "SliceSampler", "DBSCANStructureSampler"]


class BaseSampler(MSONable):
    """Base class for data sampling.

    This is for general purpose sampling without any knowledge of the data. For sampling
    methods need to know structures, see `BaseStructureSampler`.
    """

    @abc.abstractmethod
    def sample(self, data: Iterable) -> list[Any]:
        """Run the sampler to sample a subset of the data.

        Args:
            data: data to sample from.

        Returns:
            A list of sampled objects.
        """

    @property
    @abc.abstractmethod
    def indices(self) -> list[int]:
        """
        Return the indices of the data that has been sampled.
        """


class BaseStructureSampler(MSONable):
    """Base sampler from a sequence of structures."""

    @abc.abstractmethod
    def sample(self, data: list[Structure]) -> list[Structure]:
        """Run the sampler to sample a subset of the data.

        Args:
            data: data to sample from.

        Returns:
            A list of sampled structures.
        """

    @property
    @abc.abstractmethod
    def indices(self) -> list[int]:
        """
        Return the indices of the structures that has been sampled.
        """


class RandomSampler(BaseSampler):
    """Randomly sample a subset.

    Args:
        size: number of data points to sample.
        seed: random seed for the sampler.
    """

    def __init__(self, size: int, seed: int = 35):
        self.size = size
        self.seed = seed
        self._indices: list[int] = None
        np.random.seed(self.seed)

    def sample(self, data: Iterable) -> list[Any]:
        data = [x for x in data]

        if self.size > len(data):
            raise ValueError(
                f"Requested number of samples `{self.size}` larger than total "
                f"number f data points `{len(data)}`."
            )

        self._indices = [
            i for i in np.random.randint(low=0, high=len(data), size=self.size)
        ]
        selected = [data[i] for i in self._indices]

        return selected

    @property
    def indices(self) -> list[int]:
        return self._indices


class SliceSampler(BaseSampler):
    """Sample a slice of the data points.

    Args:
        index: indices of the data to select. If a list of int is provided, the
            corresponding indices are selected, e.g. `index=[0, 3, 5]` will select data
            points with indices 0, 3, and 5. Alternatively, a python slice object can be
            provided, e.g. `index=slice(10, None, 2)` will select data points with
            indices 10, 12, 14, ....
    """

    # TODO, can we not use serializable_slice? and directly use slice?
    #   This depends on how we use it in potflow.
    #   This is difficult, because self.index need to be MSONable, but slice is not.
    def __init__(self, index: list[int] | serializable_slice):
        self.index = index
        self._indices: list[int] = None

    def sample(self, data: Iterable) -> list[Any]:
        selected, self._indices = slice_sequence(data, self.index)

        return selected

    @property
    def indices(self) -> list[int]:
        return self._indices


class DBSCANStructureSampler(BaseStructureSampler):
    """Sample structures using DBSCAN clustering.

    This is achieved in the below steps:
    1. convert each structure to a list of SOAP vectors, one for each atom.
    2. (Optional) select a subset of the SOAP vectors (e.g. only the vectors for the
       atoms of a specific element).
    3. Concatenate the (selected) SOAP vectors for each structure into a single vector.
    4. (Optional) reduce the dimension of the concatenated vectors using PCA.
    5. use `sklearn.cluster.DBSCAN` to cluster the SOAP vectors. Each vector will be
       clustered into one of the following categories: `core`, `reachable`, and `noisy`.
    6. For each of the categories, randomly sample a subset of the structures.

    Args:
        soap_kwargs: arguments to pass to `dscribe.descriptors.SOAP`.
        post_soap_selection: a function to select a subset of the SOAP vectors. The
            function should take two arguments: a list of structures and a list of SOAP
            vectors, and return a list of selected SOAP vectors. If `None`, all SOAP
            vectors will be used.
        pca_dim: dimension of the PCA to perform on the concatenated SOAP vectors. If
            `None`, no PCA will be performed.
        dbscan_kwargs: arguments to pass to `sklearn.cluster.DBSCAN`.
        noisy_ratio: ratio of noisy data points to sample.
        reachable_ratio: ratio of reachable data points to sample.
        core_ratio: ratio of core data points to sample. If `auto`, the ratio will be
            set to `min_distance/average_num_neighbors`, where `min_distance` is the
            argument `min_samples` in `sklearn.cluster.DBSCAN`, and
            `average_num_neighbors` is the average number of neighbors of all core data
            points.
        seed: random seed for the sampling.
    """

    def __init__(
        self,
        soap_kwargs: dict = None,
        post_soap_selection: Callable[
            [list[Structure], list[np.ndarray]], list[np.ndarray]
        ] = None,
        pca_dim: int | None = None,
        dbscan_kwargs: dict = None,
        noisy_ratio: float = 1.0,
        reachable_ratio: float = 1.0,
        core_ratio: str | float = "auto",
        seed: int = 35,
    ):
        self.soap_kwargs = soap_kwargs if soap_kwargs is None else {}
        self.post_soap_selection = post_soap_selection
        self.pca_dim = pca_dim
        self.db_kwargs = dbscan_kwargs if dbscan_kwargs is None else {}
        self.noisy_ratio = noisy_ratio
        self.reachable_ratio = reachable_ratio
        self.core_ratio = core_ratio

        self._indices: list[int] = None

        np.random.seed(seed)

    def sample(self, data: list[Structure]) -> list[Structure]:
        """"""
        soap_vectors = self._get_soap_vectors(data)

        if self.post_soap_selection is not None:
            soap_vectors = self.post_soap_selection(data, soap_vectors)

        # TODO, need to double check whether this correct
        vectors = np.concatenate(soap_vectors, axis=1)

        if self.pca_dim is not None:
            vectors = self._dim_reduction(vectors, self.pca_dim)

        labels = self._cluster(vectors)

        if self.core_ratio == "auto":
            core_ratio = self._compute_core_ratio()
        else:
            core_ratio = self.core_ratio

        ### TODO, select core, reachable, and noisy data points based on their ratios

        # `selected` below should be a list of pymatgen Structure
        selected = []

        return selected

    def plot(self):
        """Function to plot the results of the selection.

        This can be called after the `sample` method to visualize the results.
        """

    def _get_soap_vectors(self, data: list[Structure]) -> list[np.ndarray]:
        """Convert structures to SOAP vectors."""

    def _dim_reduction(self, vectors: list[np.ndarray], dim: int):
        """Perform dimension reduction on the SOAP vectors."""

    def _cluster(self, data: list[Structure]):
        """Perform DBSCAN."""

    def _compute_core_ratio(self):
        """Compute the ratio of core data points to sample.

        ratio = min_samples/average_num_neighbors
        """
