import os

import datasets as ds
import pytest


@pytest.fixture
def dataset_path() -> str:
    return "COCOA.py"


@pytest.fixture
def data_dir() -> str:
    is_ci = bool(os.environ.get("CI", False))
    if is_ci:
        raise NotImplementedError
    else:
        return "annotations.tar.gz"


@pytest.mark.parametrize(
    argnames=(
        "dataset_name",
        "expected_num_train",
        "expected_num_validation",
        "expected_num_test",
    ),
    argvalues=(
        ("COCO", 2500, 1323, 1250),
        ("BSDS", 200, 100, 200),
    ),
)
def test_load_dataset(
    dataset_path: str,
    dataset_name: str,
    data_dir: str,
    expected_num_train: int,
    expected_num_validation: int,
    expected_num_test: int,
):
    dataset = ds.load_dataset(path=dataset_path, name=dataset_name, data_dir=data_dir)

    assert dataset["train"].num_rows == expected_num_train  # type: ignore
    assert dataset["validation"].num_rows == expected_num_validation  # type: ignore
    assert dataset["test"].num_rows == expected_num_test  # type: ignore
