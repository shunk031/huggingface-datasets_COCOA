---
language:
- en
license: cc-by-4.0

tags:
- computer-vision
- instance-segmentation
- ms-coco
- bsds

datasets:
- COCO
- BSDS

metrics:
- iou
---

# Dataset Card for COCOA

[![CI](https://github.com/shunk031/huggingface-datasets_COCOA/actions/workflows/ci.yaml/badge.svg)](https://github.com/shunk031/huggingface-datasets_COCOA/actions/workflows/ci.yaml)
<<<<<<< HEAD
<<<<<<< HEAD
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shunk031/huggingface-datasets_COCOA/blob/main/notebooks/COCOA_demo.ipynb)
=======
>>>>>>> parent of e75a659 (add notebooks (#4))
=======
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]((https://colab.research.google.com/github/shunk031/huggingface-datasets_COCOA/blob/main/notebooks/COCOA_demo.ipynb))
>>>>>>> parent of 207dbf8 (Update readme (#5))

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Dataset Preprocessing](#dataset-preprocessing)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- Homepage: https://github.com/Wakeupbuddy/amodalAPI
- Repository: https://github.com/shunk031/huggingface-datasets_COCOA
- Paper (preprint): https://arxiv.org/abs/1509.01329
- Paper (CVPR2017): https://openaccess.thecvf.com/content_cvpr_2017/html/Zhu_Semantic_Amodal_Segmentation_CVPR_2017_paper.html

### Dataset Summary

COCOA dataset targets amodal segmentation, which aims to recognize and segment objects beyond their visible parts. This dataset includes labels not only for the visible parts of objects, but also for their occluded parts hidden by other objects. This enables learning to understand the full shape and position of objects. 

From the paper:

> We propose a detailed image annotation that captures information beyond the visible pixels and requires complex reasoning about full scene structure. Specifically, we create an amodal segmentation of each image: the full extent of each region is marked, not just the visible pixels. Annotators outline and name all salient regions in the image and specify a partial depth order. The result is a rich scene structure, including visible and occluded portions of each region, figure-ground edge information, semantic labels, and object overlap. We create two datasets for semantic amodal segmentation. First, we label 500 images in the BSDS dataset with multiple annotators per image, allowing us to study the statistics of human annotations. We show that the proposed full scene annotation is surprisingly consistent between annotators, including for regions and edges. Second, we annotate 5000 images from COCO. This larger dataset allows us to explore a number of algorithmic ideas for amodal segmentation and depth ordering.

### Dataset Preprocessing

### Supported Tasks and Leaderboards

### Languages

All of annotations use English as primary language.

## Dataset Structure

### Data Instances

To use COCOA, you need to download the annotations from [the google drive](https://drive.google.com/open?id=0B8e3LNo7STslZURoTzhhMFpCelE) in the official repositories (https://github.com/Wakeupbuddy/amodalAPI#setup). Downloading of annotations currently appears to be restricted, but the author will allow us to download them if we request access privileges.

When loading a specific configuration, users has to append a version dependent suffix:

```python
import datasets as ds

dataset = ds.load_dataset("shunk031/COCOA", name="COCO", data_dir="/path/to/cocoa_annotation.tar.gz")
```

#### COCO

An example of looks as follows.

```json
{
    "image_id": 321, 
    "license_id": 1, 
    "file_name": "COCO_train2014_000000000321.jpg", 
    "height": 480, 
    "width": 640, 
    "date_captured": "2013-11-20 12: 36: 25", 
    "flickr_url": "http: //farm5.staticflickr.com/4096/4750559893_49fb0baf7f_z.jpg", 
    "image": <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x480 at 0x7FD21970F5E0>, 
    "coco_url": "http://mscoco.org/images/321", 
    "annotations": {
        "author": ["ash2"], 
        "url": ["https://s3-us-west-1.amazonaws.com/coco-ann/coco-train/COCO_train2014_000000000321.jpg"], 
        "regions": [
            {
                "segmentation": [
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=640x480 at 0x7FD21970FBE0>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=640x480 at 0x7FD21970F8E0>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=640x480 at 0x7FD21970F400>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=640x480 at 0x7FD21970F790>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=640x480 at 0x7FD21970FCA0>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=640x480 at 0x7FD21970FF40>
                ], 
                "name": ["sandwich", "container", "hot dog", "hot dog", "container", "table"], 
                "area": [63328.0, 141246.0, 31232.0, 28735.0, 265844.0, 307200.0], 
                "is_stuff": [False, False, False, False, False, True], 
                "occlude_rate": [0.0, 0.44835251569747925, 0.0, 0.022307291626930237, 0.7122523188591003, 0.9019140601158142], 
                "order": [1, 2, 3, 4, 5, 6], 
                "visible_mask": [
                    None, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=640x480 at 0x7FD21970FD90>, 
                    None, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=640x480 at 0x7FD21970FB50>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=640x480 at 0x7FD21970FE80>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=640x480 at 0x7FD219479460>
                ], 
                "invisible_mask": [
                    None, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=640x480 at 0x7FD219479160>, 
                    None, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=640x480 at 0x7FD2194793A0>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=640x480 at 0x7FD219479490>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=640x480 at 0x7FD219479130>
                ]
            }
        ], 
        "image_id": [321], 
        "depth_constraint": ["1-2,1-5,1-6,2-5,2-6,3-4,3-5,3-6,4-5,4-6,5-6"], 
        "size": [6]
    }
}
```

#### BSDS

An example of looks as follows.

```json
{
    "image_id": 100075, 
    "license_id": -100, 
    "file_name": "100075.jpg", 
    "height": 321, 
    "width": 481, 
    "date_captured": "?", 
    "flickr_url": "https://s3-us-west-1.amazonaws.com/coco-ann/BSDS/BSDS_train_100075.jpg", 
    "image": <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=481x321 at 0x7FD22A328CA0>, 
    "bsds_url": "https://s3-us-west-1.amazonaws.com/coco-ann/BSDS/BSDS_train_100075.jpg", 
    "annotations": {
        "author": ["acherian", "amorgan", "dromero", "jdayal", "kjyou", "ttouneh"], 
        "url": [
            "https://s3-us-west-1.amazonaws.com/coco-ann/BSDS/BSDS_train_100075.jpg", 
            "https://s3-us-west-1.amazonaws.com/coco-ann/BSDS/BSDS_train_100075.jpg"
        ], 
        "regions": [
            {
                "segmentation": [
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD22A3288E0>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD22A328430>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD22A328070>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD22A328610>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD22A3280D0>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD22A328BE0>
                ], 
                "name": ["rocks", "bear", "bear", "bear", "sand", "water"], 
                "area": [31872.0, 5603.0, 38819.0, 12869.0, 27883.0, 124695.0], 
                "is_stuff": [False, False, False, False, False, False], 
                "occlude_rate": [0.0, 0.0, 0.0, 0.3645193874835968, 0.13043789565563202, 0.6487349271774292], 
                "order": [1, 2, 3, 4, 5, 6], 
                "visible_mask": [
                    None, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD22A328AF0>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD22A328A30>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD22A328220>
                ], 
                "invisible_mask": [
                    None, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD22A3282E0>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD22A328400>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD22A328310>
                ]
            }, 
            {
                "segmentation": [
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD22A328340>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD22A328B80>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD22A328670>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD22A328520>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD22A328460>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD22A328D00>
                ], 
                "name": ["bear", "bear", "bear", "shore line", "water", "shore line"], 
                "area": [38772.0, 5178.0, 13575.0, 31977.0, 84224.0, 37418.0], 
                "is_stuff": [False, False, False, False, False, False], 
                "occlude_rate": [0.0, 0.0, 0.35889503359794617, 0.1458861082792282, 0.5715591907501221, 0.0], 
                "order": [1, 2, 3, 4, 5, 6], 
                "visible_mask": [
                    None, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD22A328A00>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD22A328D60>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD22A3285E0>, 
                    None
                ], 
                "invisible_mask": [
                    None, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD22A3286A0>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD22A328490>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD22A328100>, 
                    None
                ]
            }, 
            {
                "segmentation": [
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD22A3282B0>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD22A328EE0>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD22A3284C0>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD22A3285B0>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD22A328C40>
                ], 
                "name": ["bear", "bear", "bear", "beach", "ocean"], 
                "area": [38522.0, 5496.0, 12581.0, 27216.0, 126090.0], 
                "is_stuff": [False, False, False, False, False], 
                "occlude_rate": [0.0, 0.0, 0.3449646234512329, 0.11258083581924438, 0.39141881465911865], 
                "order": [1, 2, 3, 4, 5], 
                "visible_mask": [
                    None, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD22A328940>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD22A328880>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD219830A00>
                ], 
                "invisible_mask": [
                    None, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD219830CD0>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD219830BB0>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD219830940>
                ]
            }, 
            {
                "segmentation": [
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD219830910>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD2198308E0>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD219830C70>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD219830970>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD219830CA0>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD2198309A0>
                ], 
                "name": ["Bear", "Bear", "Bear", "Water", "ground", "Ground"], 
                "area": [39133.0, 7120.0, 13053.0, 97052.0, 33441.0, 26313.0], 
                "is_stuff": [False, False, False, False, False, False], 
                "occlude_rate": [0.0, 0.0, 0.4422737956047058, 0.5332708358764648, 0.007117012050002813, 0.1584388017654419], 
                "order": [1, 2, 3, 4, 5, 6], 
                "visible_mask": [
                    None, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD219830A30>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD219830C40>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD219830B80>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD2191A6820>
                ], 
                "invisible_mask": [
                    None, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD2191A68B0>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD2191A6610>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD2191A69D0>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD2191A6730>
                ]
            }, 
            {
                "segmentation": [
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD2191A6790>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD2191A6550>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD2191A6850>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD2191A6940>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD2191A66D0>
                ], 
                "name": ["bear", "bear", "bear", "water", "rock beach"], 
                "area": [38378.0, 6130.0, 12649.0, 98377.0, 153118.0], 
                "is_stuff": [False, False, False, False, False], 
                "occlude_rate": [0.0, 0.0, 0.41094157099723816, 0.5013265013694763, 0.65973299741745], 
                "order": [1, 2, 3, 4, 5], 
                "visible_mask": [
                    None, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD268700F10>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD2687004F0>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD2687002B0>
                ], 
                "invisible_mask": [
                    None, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD2191A64C0>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD28805FB50>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD28805F580>
                ]
            }, 
            {
                "segmentation": [
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD2191A6880>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD2480FB190>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD2480FB8E0>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD2480FB070>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD2480FB610>
                ], 
                "name": ["bear", "bear", "bear", "sand", "water"], 
                "area": [38802.0, 5926.0, 12248.0, 27857.0, 126748.0], 
                "is_stuff": [False, False, False, False, False], 
                "occlude_rate": [0.0, 0.0, 0.37026453018188477, 0.13170836865901947, 0.3872092664241791], 
                "order": [1, 2, 3, 4, 5], 
                "visible_mask": [
                    None, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD219479DC0>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD219479C70>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD219479A90>
                ], 
                "invisible_mask": [
                    None, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD219479AF0>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD2194795B0>, 
                    <PIL.PngImagePlugin.PngImageFile image mode=L size=481x321 at 0x7FD219479670>
                ]
            }
        ], 
        "image_id": [100075, 100075, 100075, 100075, 100075, 100075], 
        "depth_constraint": [
            "1-6,2-4,2-5,2-6,3-4,3-5,3-6,4-5,4-6,5-6", 
            "1-3,1-4,1-5,2-3,2-4,2-5,3-4,3-5,4-5", 
            "1-3,1-4,1-6,2-3,2-4,2-6,3-4,3-6,4-5,4-6", 
            "1-4,1-5,2-3,2-4,2-5,3-4,3-5,4-5", 
            "1-3,1-4,1-5,2-3,2-4,2-5,3-4,3-5,4-5"
        ], 
        "size": [6, 6, 5, 6, 5, 5]
    }
}
```

### Data Fields

#### COCO

- `image_id`: Unique numeric ID of the image.
- `license_id`: Unique numeric ID of the image license.
- `file_name`: File name of the image.
- `width`: Image width.
- `height`: Image height.
- `date_captured`: Date of capturing data
- `flickr_url`: Original flickr url of the image.
- `image`: A `PIL.Image.Image` object containing the image.
- `coco_url`: COCO url of the image.
- `annotations`: Holds a list of `Annotation` data classes:
    - `author`: TBD
    - `url`: TBD
    - `image_id`: TBD
    - `depth_constraint`: TBD
    - `size`: TBD
    - `regions`: TBD
      - `segmentation`: TBD
        - `name`: TBD
        - `area`: TBD
        - `is_stuff`: TBD
        - `occlude_rate`: TBD
        - `order`: TBD
        - `visible_mask`: TBD
        - `invisible_mask`: TBD

#### BSDS

- `image_id`: Unique numeric ID of the image.
- `license_id`: Unique numeric ID of the image license.
- `file_name`: File name of the image.
- `width`: Image width.
- `height`: Image height.
- `date_captured`: Date of capturing data
- `flickr_url`: Original flickr url of the image.
- `image`: A `PIL.Image.Image` object containing the image.
- `bsds_url`: BSDS url of the image.
- `annotations`: Holds a list of `Annotation` data classes:
    - `author`: TBD
    - `url`: TBD
    - `image_id`: TBD
    - `depth_constraint`: TBD
    - `size`: TBD
    - `regions`: TBD
      - `segmentation`: TBD
        - `name`: TBD
        - `area`: TBD
        - `is_stuff`: TBD
        - `occlude_rate`: TBD
        - `order`: TBD
        - `visible_mask`: TBD
        - `invisible_mask`: TBD

### Data Splits

| name | train | validation | test  |
|------|------:|-----------:|------:|
| COCO | 2,500 | 1,323      | 1,250 |
| BSDS | 200   | 100        | 200   |

## Dataset Creation

### Curation Rationale

### Source Data

#### Initial Data Collection and Normalization

#### Who are the source language producers?

### Annotations

#### Annotation process

#### Who are the annotators?

### Personal and Sensitive Information

## Considerations for Using the Data

### Social Impact of Dataset

### Discussion of Biases

### Other Known Limitations

## Additional Information

### Dataset Curators

### Licensing Information

COCOA is a derivative work of the COCO dataset. The authors of COCO do not in any form endorse this work. Different licenses apply:
- COCO images: [Flickr Terms of use](http://cocodataset.org/#termsofuse) 
- COCO annotations: [Creative Commons Attribution 4.0 License](http://cocodataset.org/#termsofuse)

### Citation Information

```bibtex
@inproceedings{zhu2017semantic,
  title={Semantic amodal segmentation},
  author={Zhu, Yan and Tian, Yuandong and Metaxas, Dimitris and Doll{\'a}r, Piotr},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1464--1472},
  year={2017}
}

@inproceedings{lin2014microsoft,
  title={Microsoft coco: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle={Computer Vision--ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13},
  pages={740--755},
  year={2014},
  organization={Springer}
}

@article{arbelaez2010contour,
  title={Contour detection and hierarchical image segmentation},
  author={Arbelaez, Pablo and Maire, Michael and Fowlkes, Charless and Malik, Jitendra},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={33},
  number={5},
  pages={898--916},
  year={2010},
  publisher={IEEE}
}
```

### Contributions

Thanks to [@Wakeupbuddy](https://github.com/Wakeupbuddy) for publishing the COCOA dataset.
