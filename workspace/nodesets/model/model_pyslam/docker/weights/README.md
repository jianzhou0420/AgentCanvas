# pySLAM multi-view weights (external, mounted into the container)

The `reconstruct_multiview` node's backends (MASt3R / DUSt3R / VGGT / …) need model
weights that are **not** baked into the `agentcanvas/pyslam` image. The image carries
only code + compiled CUDA kernels (`curope`); weights mount in read-only at runtime.
This keeps the image lean and lets weights update without a rebuild.

These scripts are the **tracked, canonical** copies. They download into the runtime
weights folder — `$PYSLAM_WEIGHTS_DIR`, else `<repo>/data/models/pyslam/` (which is
git-ignored, since the `.pth` files are multi-GB). That folder is what
`model_pyslam/_client.py::_weight_mounts()` bind-mounts into the container.

## Fetch (host-side, no container)

```bash
cd workspace/nodesets/model/model_pyslam/docker/weights
bash download_all.sh          # mast3r + mvdust3r explicit checkpoints
bash download_mast3r.sh       # just MASt3R (~2.5 GB) — the verified default
bash download_mvdust3r.sh     # MV-DUSt3R set (currently a known gap; see below)
```

Override the destination with `PYSLAM_WEIGHTS_DIR=/some/path bash download_mast3r.sh`.

`vggt`, `vggt_robust`, `depth_anything_v3`, `fast3r`, and pure `dust3r` fetch their
weights from HuggingFace on **first use** into a mounted `hf_cache/`, so they need no
script — the download happens once and persists.

## Layout (mirrors pyslam's `thirdparty/<backend>/checkpoints/`)

```
mast3r/checkpoints/     MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth   (explicit dl)
mvdust3r/checkpoints/   MVD.pth, MVDp_s1.pth, MVDp_s2.pth, DUSt3R_..._224_linear.pth  (explicit dl)
dust3r/checkpoints/     (pure DUSt3R — runtime HF, usually unused when MASt3R is present)
hf_cache/               HuggingFace cache for VGGT / VGGT_ROBUST / DepthAnythingV3 / Fast3r
torch_cache/            torch.hub cache (torchvision seg/depth weights)
```

## Mount contract (set by `model_pyslam/_client.py::_weight_mounts`)

```
mast3r/checkpoints   -> /home/slam/pyslam/thirdparty/mast3r/checkpoints    (ro)
mvdust3r/checkpoints -> /home/slam/pyslam/thirdparty/mvdust3r/checkpoints  (ro)
dust3r/checkpoints   -> /home/slam/pyslam/thirdparty/dust3r/checkpoints    (ro)
hf_cache             -> /home/slam/.cache/huggingface                      (rw)
torch_cache          -> /home/slam/.cache/torch                            (rw)
```

## Known gap: MV-DUSt3R

`MVDUST3R` errors at import — pySLAM's bundled mvdust3r has a version drift against its
own `dust3r` copy (`normalize_pointclouds` missing from `dust3r.utils.geometry`). The
weights are provided for completeness, but the backend is unusable until that upstream
drift is patched. MASt3R / DUSt3R / VGGT cover multi-view reconstruction today.
