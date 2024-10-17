# MOT-DETR: 3D Single Shot Detection and Tracking with Transformers to build 3D representations for Agro-Food Robots

![MOT-DETR](assets/mot-detr-arch.png "MOT-DETR")
> **MOT-DETR: 3D Single Shot Detection and Tracking with Transformers to build 3D representations for Agro-Food Robots**\
> David Rapado-Rincon, Henk Nap, Katarina Smolenova, Eldert J. van Henten and Gert Kootstra
> Paper: https://doi.org/10.1016/j.compag.2024.109275

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
@article{rapado2024mot,
  title={MOT-DETR: 3D single shot detection and tracking with transformers to build 3D representations for agro-food robots},
  author={Rapado-Rincon, David and Nap, Henk and Smolenova, Katarina and van Henten, Eldert J and Kootstra, Gert},
  journal={Computers and Electronics in Agriculture},
  volume={225},
  pages={109275},
  year={2024},
  publisher={Elsevier}
}
```
