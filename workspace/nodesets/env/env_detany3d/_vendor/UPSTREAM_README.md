> [!IMPORTANT]
> 🌟 Stay up to date at [opendrivelab.com](https://opendrivelab.com/#news)!

# DetAny3D

This is the official repository for the **[Detect Anything 3D in the Wild](https://arxiv.org/abs/2504.07958)**, a promptable 3D detection foundation model capable of detecting any novel object under arbitrary camera configurations using only monocular inputs


<!-- ## 🖼️ Demo Results

Below are example visualizations of DetAny3D predictions:

<p align="center">
  <img src="assets/demo1.jpg" alt="Demo 1" width="400"/>
  <img src="assets/demo2.jpg" alt="Demo 2" width="400"/>
</p>

<p align="center">
  <img src="assets/demo3.jpg" alt="Demo 3" width="400"/>
  <img src="assets/demo4.jpg" alt="Demo 4" width="400"/>
</p> -->

## 📖 Table of Contents

- [📌 TODO](#-todo)
- [🚀 Getting Started](#-getting-started)
  - [Step 1: Create Environment](#step-1-create-environment)
  - [Step 2: Install Dependencies](#step-2-install-dependencies)
- [📦 Checkpoints](#-checkpoints)
- [📁 Dataset Preparation](#-dataset-preparation)
- [🏋️‍♂️ Training](#️-training)
- [🔍 Inference](#-inference)
- [🌐 Launch Online Demo](#-launch-online-demo)
- [📚 Citation](#-citation)


## 📌 TODO

### ✅ Done
- Release full code
- Provide training and inference scripts
- Release the model weights

### 🛠️ In Progress
- **TODO**: Provide full conversion scripts for constructing DA3D locally
- **TODO**: Simplify the inference process
- **TODO**: Provide a tutorial for creating customized datasets and finetuning


## 🚀 Getting Started

### Step 1: Create Environment

```
conda create -n detany3d python=3.8
conda activate detany3d
```

---

### Step 2: Install Dependencies

#### ✅ (1) Install [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)

Follow the official instructions to install SAM and download its checkpoints.

#### ✅ (2) Install [UniDepth](https://github.com/lpiccinelli-eth/UniDepth)

Follow the UniDepth setup guide to compile and install all necessary packages.

#### ✅ (3) Clone and configure [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

```
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
```

> 👉 The exact dependency versions are listed in our `requirements.txt`


## 📦 Checkpoints

Please download third-party checkpoints from the following sources:

- **SAM checkpoint**: Please download `sam_vit_h.pth` from the official [SAM GitHub Releases](https://github.com/facebookresearch/segment-anything)
- **UniDepth / DINO checkpoints**: Available via [Google Drive](https://drive.google.com/drive/folders/17AOq5i1pCTxYzyqb1zbVevPy5jAXdNho?usp=drive_link)

```
detany3d_private/
├── checkpoints/
│   ├── sam_ckpts/
│   │   └── sam_vit_h.pth
│   ├── unidepth_ckpts/
│   │   └── unidepth.pth
│   ├── dino_ckpts/
│   │   └── dino_swin_large.pth
│   └── detany3d_ckpts/
│       └── detany3d.pth
```

> GroundingDINO's checkpoint should be downloaded from its [official repo](https://github.com/IDEA-Research/GroundingDINO) and placed as instructed in their documentation.


## 📁 Dataset Preparation

The `data/` directory should follow the structure below:

```
data/
├── DA3D_pkls/                             # DA3D processed pickle files 
├── kitti/
│   ├── test_depth_front/
│   ├── ImageSets/
│   ├── training/
│   └── testing/
├── nuscenes/
|   ├── nuscenes_depth/
│   └── samples/
├── 3RScan/
│   └── <token folders>/             # e.g., 10b17940-3938-...
├── hypersim/
|   ├── depth_in_meter/
│   └── ai_XXX_YYY/                  # e.g., ai_055_009
├── waymo/
│   └── kitti_format/                # KITTI-format data for Waymo
│       ├── validation_depth_front/
│       ├── ImageSets/
│       ├── training/
│       └── testing/
├── objectron/
│   ├── train/
│   └── test/
├── ARKitScenes/
│   ├── Training/
│   └── Validation/
├── cityscapes3d/
│   ├── depth/
│   └── leftImg8bit/
├── SUNRGBD/
│   ├── realsense/
│   ├── xtion/
|   ├── kv1/
│   └── kv2/
```

> The download for `kitti`, `nuscenes`, `hypersim`, `objectron`, `arkitscenes`, and `sunrgbd` follow the [Omni3D](https://github.com/facebookresearch/omni3d) convention. Please refer to the Omni3D repository for details on how to organize and preprocess these datasets.

> 🗂️ The `DA3D_pkls` (minimal metadata for inference) can be downloaded from [Google Drive](https://drive.google.com/drive/folders/17AOq5i1pCTxYzyqb1zbVevPy5jAXdNho?usp=drive_link).  
> 🧩 **Note**: This release currently supports a minimal inference-only version. The conversion scripts of full dataset + all depth-related files will be provided later.

> ⚠️ Depth files are not required for inference. You can safely set `depth_path = None` in [detany3d_dataset.py](./detect_anything/datasets/detany3d_dataset.py) to bypass depth loading.  



## 🏋️‍♂️ Training

```
torchrun \
    --nproc_per_node=8 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nnodes=8 \
    --node_rank=${RANK} \
    ./train.py \
    --config_path \
    ./detect_anything/configs/train.yaml
```


## 🔍 Inference

```
torchrun \
    --nproc_per_node=8 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nnodes=1 \
    --node_rank=${RANK} \
    ./train.py \
    --config_path \
    ./detect_anything/configs/inference_indomain_gt_prompt.yaml
```


After inference, a file named `{dataset}_output_results.json` will be generated in the `exps/<your_exp_dir>/` directory.

> ⚠️ Due to compatibility issues between `pytorch3d` and the current environment, we recommend copying the output JSON file into the evaluation script of repositories like [Omni3D](https://github.com/facebookresearch/omni3d) or [OVMono3D](https://github.com/UVA-Computer-Vision-Lab/ovmono3d) for standardized metric evaluation.

> **TODO**: Evaluation for zero-shot datasets currently requires manual modification of the Omni3D or OVMono3D repositories and is not yet fully supported here.   
We plan to release a merged evaluation script in this repository to make direct evaluation more convenient in the future.



## 🌐 Launch Online Demo

```
python ./deploy.py
```


## 📚 Citation

If you find this repository useful, please consider citing:

```
@article{zhang2025detect,
  title={Detect Anything 3D in the Wild},
  author={Zhang, Hanxue and Jiang, Haoran and Yao, Qingsong and Sun, Yanan and Zhang, Renrui and Zhao, Hao and Li, Hongyang and Zhu, Hongzi and Yang, Zetong},
  journal={arXiv preprint arXiv:2504.07958},
  year={2025}
}
```
