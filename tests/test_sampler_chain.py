from potdata.sampler_chain import SamplerChain
from potdata.samplers import SliceSampler


def test_sampler_chain():
    sampler = SliceSampler(slice(2, 20, 2))

    sc = SamplerChain(samplers=[sampler, sampler])

    data = list(range(20))

    sampled_data = sc.sample(data)

    assert sampled_data == [6, 10, 14, 18]
    assert sc.indices == [6, 10, 14, 18]
