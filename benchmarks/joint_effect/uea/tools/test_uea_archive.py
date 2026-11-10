import numpy as np


def test_archive_preserves_explicit_train_test_splits(tmp_path):
    from zipfile import ZipFile

    from benchmarks.joint_effect.uea.tools.archive import archive_split

    header = "@timestamps false\n@classlabel true a b\n@data\n"
    with ZipFile(tmp_path / "BasicMotions.zip", "w") as archive:
        archive.writestr(
            "nested/BasicMotions_TRAIN.ts", header + "1,2:3,4:a\n2,3:4,5:b\n"
        )
        archive.writestr(
            "nested/BasicMotions_TEST.ts", header + "8,9:2,1:b\n7,8:0,1:a\n"
        )
    train, y_train, test, y_test = archive_split("BasicMotions", tmp_path)
    np.testing.assert_array_equal(train[0], [[1, 3], [2, 4]])
    np.testing.assert_array_equal(test[0], [[8, 2], [9, 1]])
    assert y_train.tolist() == ["a", "b"]
    assert y_test.tolist() == ["b", "a"]
