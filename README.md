# MOT-DETR: 3D Single Shot Detection and Tracking with Transformers to build 3D representations for Agro-Food Robots

![MOT-DETR](assets/mot-detr-arch.png "MOT-DETR")
> **MOT-DETR: 3D Single Shot Detection and Tracking with Transformers to build 3D representations for Agro-Food Robots**\
> David Rapado-Rincon, Henk Nap, Katarina Smolenova, Eldert J. van Henten and Gert Kootstra
> Paper: https://arxiv.org/abs/2311.15674

## About
Official implementation of the paper *MOT-DETR: 3D Single Shot Detection and Tracking with Transformers to build 3D representations for Agro-Food Robots*. Work submitted to ICRA 2024.

It contains the original models and code to train MOT-DETR in your custom dataset.

## Train on custom dataset
We provide the original dataset used in our experiments with our dataset. You can see it here: [dataset.py](datasets/dataset.py). You can develop your custom dataset for your own data by using it as a template. The dataset needs to output the following:
- RGB image tensor (3, H, W)
- XYZ organized pointcloud tensor where NaNs are set to zero (3, H, W)
- A dictionary containing the labels with the following keys:
    - `labels` - class of the objects (M, 1), where M is the total maximum of object present in a single image in the dataset. We use 33 (see line 170 at [dataset.py](datasets/dataset.py)).
    - `boxes` - bounding boxes of the objects (M, 4) in the format (cx, cy, w, h), where M is the total maximum of object present in a single image in the dataset. We use 33 (see line 170 at [dataset.py](datasets/dataset.py)).
    - `track_labels` - As the tracking head is trained as a classification task where each object ID corresponds to a single class, this dict contains the track class of the objects (M, 1), where M is the total maximum of object present in a single image in the dataset. We use 33 (see line 170 at [dataset.py](datasets/dataset.py))



## Citation
```
@misc{rapadorincon2023motdetr,
      title={MOT-DETR: 3D Single Shot Detection and Tracking with Transformers to build 3D representations for Agro-Food Robots}, 
      author={David Rapado-Rincon and Henk Nap and Katarina Smolenova and Eldert J. van Henten and Gert Kootstra},
      year={2023},
      eprint={2311.15674},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```