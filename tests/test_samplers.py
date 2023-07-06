from potdata.samplers import RandomSampler, SliceSampler


def test_random_sampler():
    size = 5
    sampler = RandomSampler(size)
    data = list(range(10))
    indices = data
    sampled_data = sampler.sample(data)

    assert len(sampled_data) == size
    assert all([i in data for i in sampled_data])

    sampled_indices = sampler.indices
    assert len(sampled_indices) == size
    assert all([i in indices for i in sampled_indices])

    assert sampled_data == sampled_indices


def test_slice_sampler():
    sampler = SliceSampler(slice(2, 8, 2))
    data = list(range(10))
    sampled_data = sampler.sample(data)

    assert sampled_data == [2, 4, 6]

    sampled_indices = sampler.indices
    assert sampled_indices == [2, 4, 6]
