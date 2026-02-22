# Light4D: Training-Free Extreme Viewpoint 4D Video Relighting

Light4D is a novel training-free framework designed to synthesize consistent 4D videos under target illumination, even under extreme viewpoint changes.

<div align="center">
  <a href="https://scholar.google.com/citations?user=zSlYF4QAAAAJ&hl=zh-CN" target="_blank"><strong>Zhenghuang Wu*</strong></a>&emsp;
  <a href="https://scholar.google.com/citations?hl=en&user=yvWoPjAAAAAJ" target="_blank"><strong>Kang Chen*</strong></a>&emsp;
  <a href="https://steve-zeyu-zhang.github.io/" target="_blank"><strong>Zeyu Zhang†</strong></a>&emsp;
  <a href="https://ha0tang.github.io/" target="_blank"><strong>Hao Tang‡</strong></a>
</div>

<div align="center">
  School of Computer Science, Peking University
</div>

<div align="center">
  *Equal contribution. †Project lead. ‡Corresponding author: bjdxtanghao@gmail.com.
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2602.11769" target='_blank'>
    <img src="http://img.shields.io/badge/arXiv-2602.11769-b31b1b?logo=arxiv&logoColor=b31b1b" alt="ArXiv" style="height:20px; vertical-align:middle;">
  </a>
  <a href="https://aigeeksgroup.github.io/Light4D/" target='_blank'>
    <img src="https://img.shields.io/badge/Project-Page-red?logo=googlechrome&logoColor=red" alt="Project Page" style="height:20px; vertical-align:middle;">
  </a>
</p>


## 🔧 Installation
### Clone Repository
  ```
  git clone https://github.com/AIGeeksGroup/Light4D
  cd Light4D
  ```
### Setup Environment
  ```
  conda create -n Light4D python=3.10.19
  conda activate Light4D

  pip install -r requirements.txt

  # Install Nvdiffrast
  pip install git+https://github.com/NVlabs/nvdiffrast.git
  # Install dependencies and diffsynth
  pip install -e .
  # Install depthcrafter. (Follow DepthCrafter's installing instruction for checkpoints preparation.)
  git clone https://github.com/Tencent/DepthCrafter.git
  ```

## 🔑 Download Pretrained Models

```bash
hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./models/Wan-AI
hf download yihouxiang/EX-4D --local-dir ./models/EX-4D
hf download lllyasviel/ic-light iclight_sd15_fc.safetensors --local-dir ./models
hf download stablediffusionapi/realistic-vision-v51 --local-dir ./models/realistic-vision-v51
```


## 🚀 Quick Start

### Reconstruct the Source Video
```bash
bash recon.sh
```
This step reconstructs a rendered source sequence and writes the inputs required by relighting.

Expected outputs (under the `--output_dir` used in `recon.sh`):
- `color_<cam>.mp4`: RGB conditioning video
- `mask_<cam>.mp4`: binary visibility mask video
- `camera_<cam>.npz` (optional): camera extrinsics/intrinsics when `--save_camera` is enabled

For the default `recon.sh` settings (`--cam 30`, `--output_dir outputs/bear/cam30`), the generated files are:
- `outputs/bear/cam30/color_30.mp4`
- `outputs/bear/cam30/mask_30.mp4`

Use these two MP4 files as `--color_video` and `--mask_video` in the relighting command below.

### Generate a 4D Relighting Video
```bash
python 4D_relighting.py \
    --color_video "/path/to/color.mp4" \
    --mask_video "/path/to/mask.mp4" \
    --output_video "/outputs/relighting.mp4" \
    --sd_model "models/realistic-vision-v51" \
    --ic_light_model "models/iclight_sd15_fc.safetensors" \
    --enable_relight \
    --light_direction LEFT \
    --relight_prompt "Natural sunlight, cinematic quality" \
    --gamma 0.7 \
    --seed 42 \
    --num_inference_steps 30 \
    --num_frames 49 \
    --height 384 \
    --width 384 \
    --device "cuda" \
    --use_progressive_fusion
```
  `--light_direction` supports `LEFT`, `RIGHT`, and `TOP`.

## 📚 Citation
If you find our work useful for your research, please consider giving us a star and citing our paper:

```
@article{wu2026light4d,
  title={Light4D: Training-Free Extreme Viewpoint 4D Video Relighting},
  author={Wu, Zhenghuang and Chen, Kang and Zhang, Zeyu and Tang, Hao},
  journal={arXiv preprint arXiv:2602.11769},
  year={2026}
}
```

## ♥️ Acknowledgement

We sincerely thank the authors and open-source contributors of [EX-4D](https://github.com/tau-yihouxiang/EX-4D) and [IC-Light](https://github.com/lllyasviel/IC-Light).
Their excellent work and released resources provide an important foundation for our research and implementation.

