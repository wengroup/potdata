from potdata.io.adaptor import (
    ACECollectionAdaptor,
    ExtxyzAdaptor,
    ExtxyzCollectionAdaptor,
    MTPCollectionAdaptor,
    VasprunAdaptor,
    VasprunCollectionAdaptor,
)
from potdata.utils.dataops import set_field_precision, set_field_to_none
from potdata.utils.path import to_path


# NOTE, tmpdir is a pytest builtin fixture
def test_vasprun_adaptor(tmpdir, test_data_dir):
    vasprun = test_data_dir / "vasp/Si_double_relax/relax_2/outputs/vasprun.xml.gz"

    with tmpdir.as_cwd():
        adaptor = VasprunAdaptor()
        datapoints = adaptor.read(vasprun)

    assert len(datapoints) == 1


def test_vasprun_collection_adaptor(tmpdir, test_data_dir):
    path = test_data_dir / "vasp/Si_double_relax"

    with tmpdir.as_cwd():
        adaptor = VasprunCollectionAdaptor()
        datapoints = adaptor.read(path)

    assert len(datapoints) == 2


def test_extxyz_adaptor(relax_fitting_data_points, tmpdir):
    with tmpdir.as_cwd():
        adaptor = ExtxyzAdaptor()

        filename = "config.xyz"

        dp_write = relax_fitting_data_points[0]
        adaptor.write(dp_write, filename, reference_energy=None)

        dp_read = adaptor.read(filename)

        _compare_two_data_points(dp_write, dp_read)


def test_extxyz_collection_adaptor(relax_fitting_data_collection, tmpdir):
    with tmpdir.as_cwd():
        adaptor = ExtxyzCollectionAdaptor()

        path = "xyz_collection"

        _ = adaptor.write(relax_fitting_data_collection, path, reference_energy=None)

        all_dp_read_by_name = {dp.label: dp for dp in adaptor.read(path).data_points}

        # sort the data points by the index in the label
        all_dp_read_by_name = {
            int(to_path(k).stem.split("-")[1]): v
            for k, v in all_dp_read_by_name.items()
        }
        all_dp_read = [
            all_dp_read_by_name[i] for i in sorted(all_dp_read_by_name.keys())
        ]

        for dp_write, dp_read in zip(
            relax_fitting_data_collection.data_points, all_dp_read
        ):
            _compare_two_data_points(dp_write, dp_read)


def test_ace_collection_adaptor(relax_fitting_data_collection, tmpdir):
    with tmpdir.as_cwd():
        adaptor = ACECollectionAdaptor()
        filename = "ace_data.pkl.gzip"

        adaptor.write(relax_fitting_data_collection, filename, reference_energy=None)
        all_dp_read = adaptor.read(filename)

        for dp_write, dp_read in zip(
            relax_fitting_data_collection.data_points, all_dp_read.data_points
        ):
            _compare_two_data_points(
                dp_write,
                dp_read,
                fields_to_remove=("uuid", "provenance"),
                property_fields_to_remove=("stress",),
                property_to_set_precision=dict(),
            )


def test_mtp_collection_adaptor(relax_fitting_data_collection, tmpdir):
    adaptor = MTPCollectionAdaptor()

    coords = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
    assert adaptor._get_min_dist(coords) == 2**0.5

    # TODO, add more tests
    with tmpdir.as_cwd():
        filename = "mtp_data.cfg"
        adaptor.write(relax_fitting_data_collection, filename, reference_energy=None)


def _compare_two_data_points(
    dp1,
    dp2,
    fields_to_remove=("provenance", "uuid"),
    property_fields_to_remove=(),
    property_to_set_precision={"stress": 10},
):
    # remove certain fields and set the precision of some fields for comparison
    dp1 = set_field_to_none(dp1, fields=fields_to_remove)
    dp2 = set_field_to_none(dp2, fields=fields_to_remove)

    for name in property_fields_to_remove:
        dp1 = set_field_to_none(dp1.property, fields=[name])
        dp2 = set_field_to_none(dp2.property, fields=[name])

    # set the precision of some fields for comparison
    for k, v in property_to_set_precision.items():
        dp1 = set_field_precision(dp1.property, fields=[k], digits=v)
        dp2 = set_field_precision(dp2.property, fields=[k], digits=v)

    assert dp1 == dp2
