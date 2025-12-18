from potdata.io.adaptor import MTPCollectionAdaptor


def test_mtp_adaptor(tmpdir, test_data_dir):

    adaptor = MTPCollectionAdaptor()
    coords = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
    assert adaptor._get_min_dist(coords) == 2**0.5

    # read
    filename = test_data_dir / "io" / "mtp.cfg"
    dc = adaptor.read(filename, type_map={0: "Al", 1: "Fe"})
    assert len(dc) == 2

    # write
    with tmpdir.as_cwd():
        filename = "mtp_data.cfg"
        adaptor.write(dc, filename, reference_energy=None)

        dc2 = adaptor.read(filename, type_map={0: "Al", 1: "Fe"})
