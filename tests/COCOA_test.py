import datasets as ds
import pytest


@pytest.fixture
def dataset_path() -> str:
    return "COCOA.py"


@pytest.mark.parametrize(
    argnames="dataset_name",
    argvalues=(
        "COCO",
        # "BSDS",
    ),
)
def test_load_dataset(
    dataset_path: str, dataset_name: str, data_dir: str = "./annotations.tar.gz"
):
    dataset = ds.load_dataset(path=dataset_path, name=dataset_name, data_dir=data_dir)
