# Retrieval-Augmented Manipulation

This repository provides the implementation of a retrieval-augmented framework that enables VLM spatial awareness for object-centric robot manipulation.

## Repository Layout

```text
config/                     Project configuration files
controller/                 Robot, camera, and viewer interfaces
languages/                  VLM prompts and OpenAI-compatible client
planner/                    Constraint parsing, path planning, and trajectory generation
tools/ram_training/         Offline RAM training entry point
tools/synthetic_data_renderer/
                            BlenderProc pipeline for synthetic RAM training data
tools/template_annotator/   Interactive primitive annotation tool
tools/template_renderer/    Template rendering
utils/                      Shared geometry, image, configuration, and sensor utilities
visions/grounding/          GroundingDINO wrapper
visions/ram/                RAM inference, template metadata, and utilities
visions/vggt/               VGGT module used by the RAM step
```

## Setup

### System Requirements

The project has been tested with the following environment:

- Ubuntu 22.04
- Python 3.10
- CUDA 12.8

Please install the Python dependencies with:

```bash
pip install -r requirements.txt
```

Some dependencies are platform-specific and are therefore not installed by `requirements.txt`:

- Fairino robot SDK, which provides the `Robot` module used by `controller/fairino_arm.py`.
- ZED SDK Python bindings, required by `ZEDCamera`.
- RealSense Python bindings, required by D435/D455 cameras.
- Model checkpoints for GroundingDINO/SAM2, DINOv2, and VGGT.
- Grounded-SAM-2, used by `visions/grounding/dino_grounding.py`. Please follow the original [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) repository and install/configure it under `visions/grounding/Grounded-SAM-2`.

Please edit [config/config.yaml](config/config.yaml) to match your local paths, devices, robot IP, camera calibration files, checkpoints, and RAM template location.

## Synthetic Training Data Generation

Please first install [BlenderProc](https://github.com/DLR-RM/BlenderProc/tree/main), which will be used by the synthetic data and template rendering tools.

Then, synthetic RAM training scenes can be generated with BlenderProc:

```bash
cd tools/synthetic_data_renderer
blenderproc run render_ram_training_scenes.py \
  --num-scenes 400 \
  --views-per-scene 25
```

This command reads textures from `assets/cc_textures/`, CAD metadata and meshes from `bop_datasets/ram/models/`, and writes BOP-style output under `outputs/training_scenes/bop_data/ram/`. Please download [cc_textures](https://gocuhk-my.sharepoint.com/:u:/g/personal/kaichen_cuhk_edu_hk/IQDKwAASO4XCSp9xlQcvgED_AbHDRVn4Lc9o2gq9NTQ69oQ?e=MzSOO8) and [models](https://gocuhk-my.sharepoint.com/:u:/g/personal/kaichen_cuhk_edu_hk/IQCEY11EZwtFSJni0OehvLQ8AacgKvg644gSwBTeGpd9keE?e=IUomBe), and move them to the corresponding folder before synthetic data generation.

To render a subset of objects, please use `--object-ids`:

```bash
blenderproc run render_ram_training_scenes.py \
  --object-ids 501 502 503 \
  --num-scenes 20
```

## RAM Training

The RAM training entry point is:

```bash
python tools/ram_training/train_bop.py \
  --data-dir tools/ram_training/data/BOP \
  --disable-wandb
```

The expected training data layout is:

```text
tools/ram_training/data/BOP/
  bop_train_list.txt
  bop_test_list.txt
  bop_cad_model.pkl
  cad_models/model_meta.json     # or models/model_meta.json
  dinov2_b14_template/
    <template_id>.pkl
```

The rendered samples referenced by the list files should contain BOP-style `scene_gt.json`, `rgb/`, `depth/`, `mask/`, and `nocs/` data.

Training uses `RAMNet` with DINOv2-B/14 and freezes the DINO backbone.

## Template Preparation

For the template of a target object category, its template views can be rendered with BlenderProc, following a similar process with synthetic data generation.

```bash
blenderproc run tools/template_renderer/render_templates.py \
  --output-dir outputs/template_renderer \
  --category bowl
```

The rendered data can then be converted into the final RAM foundation `.pkl` file:

```bash
python tools/template_renderer/process_bop_templates.py \
  --bop-root outputs/template_renderer/bop_data/ram_template \
  --category bowl \
  --object-id 561 \
  --template-id 561 \
  --dinov2-checkpoint /path/to/dinov2_vitb14_pretrain.pth \
  --copy-mesh
```

The default output layout is:

```text
visions/ram/template/objects/bowl/000561/
  obj_000561.ply
  000561.pkl
```

Please use `--save-source` only when debugging intermediate cropped views and extracted DINOv2 feature maps.

## Template Annotation

RAM primitives for each object category can be annotated on a mesh with:

```bash
python tools/template_annotator/annotator.py
```

For a custom mesh, please specify the mesh and output directory:

```bash
python tools/template_annotator/annotator.py \
  --mesh /path/to/object.ply \
  --output-dir visions/ram/template/annotated \
  --scale 0.002 \
  --port 8080
```

The annotator exports primitive metadata such as `contact_point_*`, `plane_*`, `sliding_direction`, `rotation_axis_direction`, and `rotation_axis_point`.

## Other Assets

Please refer to the upstream projects for installation and configuration of the external components used by this repository:

- [GroundingDINO](https://github.com/idea-research/groundingdino), used by the visual grounding adapter.
- [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2), used by `visions/grounding/dino_grounding.py`.
- [VGGT](https://github.com/facebookresearch/vggt), used by the RAM step for point cloud reconstruction.

Large assets are intentionally not committed to this repository. Please download or prepare them separately before running the corresponding components.

The expected checkpoint locations are:

```text
checkpoints/
  sam2.1_hiera_large.pt
  groundingdino_swint_ogc.pth

visions/ram/checkpoints/
  dinov2_vitb14_pretrain_torch1.pth
```

RAM runtime template assets should follow the layout below:

```text
visions/ram/template/objects/
  <category>/
    <template_id>/
      obj_<template_id>.ply
      <template_id>.json
      <template_id>.pkl
```

All these third-party components, model checkpoints, datasets, CAD meshes, and generated assets are governed by their own licenses. Please review and comply with the corresponding upstream license terms before using or redistributing them.

## Retrieval-Augmented Manipulation

The runtime pipeline for the manipulation framework is organized into four main stages:

```text
step1_grounding.py   # capture images, perform VLM object discovery, and run 2D visual grounding
step2_ram.py         # estimate object poses, coordinate maps, grasp points, and functional planes
step3_planning.py    # decompose the task into VLM action constraints
step4_conducting.py  # parse VLM actions, generate trajectories, and execute them with the robot
```

## Citation

If you find this project useful, please consider citing:

```bibtex
@article{chen2026retrieval,
  title={A retrieval-augmented framework enabling VLM spatial awareness for object-centric robot manipulation},
  author={Chen, Kai and Li, Chengkun and Tu, Chang and Pan, Jiahui and Ma, Yiyao and Chen, Wei and Zhou, Zhongxiang and Xu, Xuecheng and James, Stephen and Fu, Chi-Wing and Xiong, Rong and Abbeel, Pieter and Liu, Yun-Hui and Dou, Qi},
  journal={Science Robotics},
  year={2026}
}
```
