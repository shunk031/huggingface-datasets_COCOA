import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import datasets as ds
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
JsonDict = Dict[str, Any]

ImageId = int
AnnotationId = int
LicenseId = int


_CITATION = """\
@inproceedings{zhu2017semantic,
  title={Semantic amodal segmentation},
  author={Zhu, Yan and Tian, Yuandong and Metaxas, Dimitris and Doll{\'a}r, Piotr},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1464--1472},
  year={2017}
}
"""

_DESCRIPTION = """\

"""

_HOMEPAGE = "https://github.com/Wakeupbuddy/amodalAPI"

_LICENSE = """\
TBD
"""


@dataclass
class ImageData(object):
    image_id: ImageId
    license_id: LicenseId
    file_name: str
    coco_url: str
    height: int
    width: int
    date_captured: str
    flickr_url: str

    @classmethod
    def from_dict(cls, json_dict: JsonDict) -> "ImageData":
        return cls(
            image_id=json_dict["id"],
            license_id=json_dict["license"],
            file_name=json_dict["file_name"],
            coco_url=json_dict["coco_url"],
            height=json_dict["height"],
            width=json_dict["width"],
            date_captured=json_dict["date_captured"],
            flickr_url=json_dict["flickr_url"],
        )

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)


@dataclass
class CocoaAnnotationData(object):
    author: str
    url: str
    regions: str
    image_id: ImageId
    depth_constraint: str
    annotation_id: AnnotationId
    size: int


def _load_images_data(
    image_dicts: List[JsonDict],
    tqdm_desc: str = "Load images",
) -> Dict[ImageId, ImageData]:
    images = {}
    for image_dict in tqdm(image_dicts, desc=tqdm_desc):
        image_data = ImageData.from_dict(image_dict)
        images[image_data.image_id] = image_data
    return images


def _load_cocoa_data(
    ann_dicts: List[JsonDict],
    images: Dict[ImageId, ImageData],
    tqdm_desc: str = "Load COCOA annotations",
):
    annotations = defaultdict(list)
    ann_dicts = sorted(ann_dicts, key=lambda d: d["image_id"])

    for ann_dict in tqdm(ann_dicts, desc=tqdm_desc):
        cocoa_data = CocoaAnnotationData.from_dict(ann_dict, images=images)
        annotations[cocoa_data.image_id].append(cocoa_data)

    return annotations


class CocoaDataset(ds.GeneratorBasedBuilder):
    VERSION = ds.Version("1.0.0")
    BUILDER_CONFIGS = [
        ds.BuilderConfig(name="COCO", version=VERSION),
        ds.BuilderConfig(name="BSDS", version=VERSION),
    ]

    def load_amodal_annotation(self, ann_json_path: str) -> JsonDict:
        logger.info(f"Load from {ann_json_path}")
        with open(ann_json_path, "r") as rf:
            ann_json = json.load(rf)
        return ann_json

    @property
    def manual_download_instructions(self) -> str:
        return "TBD"

    def _info(self) -> ds.DatasetInfo:
        features = ds.Features()
        return ds.DatasetInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            features=features,
        )

    def _split_generators(self, dl_manager: ds.DownloadManager):
        assert dl_manager.manual_dir is not None, dl_manager.manual_dir
        data_path = os.path.expanduser(dl_manager.manual_dir)

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"{data_path} does not exists. Make sure you insert a manual dir "
                'via `datasets.load_dataset("shunk031/COCOA", data_dir=...)` '
                "that includes tar/untar files from the COCOA annotation tar.gz. "
                f"Manual download instructions: {self.manual_download_instructions}"
            )
        else:
            data_path = (
                dl_manager.extract(data_path)
                if not os.path.isdir(data_path)
                else data_path
            )

        assert isinstance(data_path, str)
        ann_dir = os.path.join(data_path, "annotations")

        if self.config.name == "COCO":
            tng_ann_path = os.path.join(
                ann_dir,
                f"{self.config.name}_amodal_train2014.json",
            )
            val_ann_path = os.path.join(
                ann_dir,
                f"{self.config.name}_amodal_val2014.json",
            )
            tst_ann_path = os.path.join(
                ann_dir,
                f"{self.config.name}_amodal_test2014.json",
            )
        elif self.config.name == "BSDS":
            tng_ann_path = os.path.join(
                ann_dir,
                f"{self.config.name}_amodal_train.json",
            )
            val_ann_path = os.path.join(
                ann_dir,
                f"{self.config.name}_amodal_val.json",
            )
            tst_ann_path = os.path.join(
                ann_dir,
                f"{self.config.name}_amodal_test.json",
            )
        else:
            raise ValueError(f"Invalid name: {self.config.name}")

        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,
                gen_kwargs={"amodal_annotation_path": tng_ann_path},
            ),
            ds.SplitGenerator(
                name=ds.Split.VALIDATION,
                gen_kwargs={"amodal_annotation_path": val_ann_path},
            ),
            ds.SplitGenerator(
                name=ds.Split.TEST, gen_kwargs={"amodal_annotation_path": tst_ann_path}
            ),
        ]

    def _generate_examples(self, amodal_annotation_path: str):
        ann_json = self.load_amodal_annotation(amodal_annotation_path)

        images = _load_images_data(image_dicts=ann_json["images"])
        annotations = _load_cocoa_data(ann_dicts=ann_json["annotations"], images=images)

        breakpoint()
