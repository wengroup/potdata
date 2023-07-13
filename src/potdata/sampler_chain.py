"""This module contains classes for chaining multiple samplers."""

from typing import Any, Sequence

from potdata.samplers import BaseSampler


class SamplerChain(BaseSampler):
    """Chaining multiple samplers and treat them as a single sampler.

    The samplers are applied sequentially. The input data is fed to the first sampler,
    and its output data is fed as the input for the second sampler, and so on.
    """

    def __init__(self, samplers: list[BaseSampler]):
        self.samplers = samplers
        self._indices: list[int] = []

    def sample(self, data: Sequence[Any]) -> list[Any]:
        """
        Apply the sampler to a list of data.

        Args:
            data: Data to sample.

        Returns:
            A list of sampled data.
        """
        selected_indices = list(range(len(data)))
        for sampler in self.samplers:
            data = sampler.sample(data)
            selected_indices = [selected_indices[i] for i in sampler.indices]

        self._indices = selected_indices

        return list(data)

    @property
    def indices(self) -> list[int]:
        """
        Indices of the data sampled by the sampler.

        Returns:
            A list of indices.
        """
        return self._indices
