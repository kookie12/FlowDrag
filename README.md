<p align="center">
  <h2 align="center"><strong>FlowDrag:<br> 3D-aware Drag-based Image Editing with <br> Mesh-guided Deformation Vector Flow Fields</strong></h2>

<p align="center">
    <a href="https://kookie12.github.io/">Gwanhyeong Koo</a>,
    <a href="https://dbstjswo505.github.io/">Sunjae Yoon</a>,
    <a href="http://sanctusfactory.com/family_02.php">Younghwan Lee</a>,
    <a href="https://jiwoohong93.github.io/">Ji Woo Hong</a>,
    <a href="http://sanctusfactory.com/family.php">Chang D. Yoo</a>
    <br>
    <b>KAIST</b>
</p>

## 📢 Release  
- **[07/12/2024]** Initial preview release
- **[12/28/2024]** Code and VFD-Bench dataset released

## 🐶 Introduction

**FlowDrag** is a 3D-aware drag-based image editing method that leverages mesh-guided deformation vector flow fields. Our approach generates spatially coherent edits by utilizing 3D mesh deformations to guide the flow field.

### Key Features
- 🎯 **3D-Aware Editing**: Utilizes 3D mesh deformations for spatially coherent edits
- 🌊 **Flow Field Guidance**: Generates dense deformation vector fields from sparse user inputs
- 📊 **VFD-Bench Dataset**: Comprehensive benchmark for evaluating drag-based editing methods
- 🚀 **Interactive UI**: User-friendly Gradio interface for real-time editing

## 📁 Project Structure

```
FlowDrag/
├── flowdrag_ui.py              # Main inference UI
├── pipeline.py                 # FlowDrag pipeline implementation
├── bench_flowdrag.py           # Evaluation on VFD-Bench
├── dataset/
│   └── VFD_Bench_Dataset/      # Benchmark dataset
├── mesh_deformation/
│   └── flowdrag_mesh_deform_ui.py  # Mesh deformation UI
├── utils/
│   ├── drag_utils.py           # Core drag editing utilities
│   ├── lora_utils.py           # LoRA training utilities
│   ├── attn_utils.py           # Attention manipulation
│   └── ui_utils.py             # UI helper functions
├── samples/                    # Sample images and results
└── environment_flowdrag.yaml   # Conda environment
```

## 💻 Installation

### Setup

```bash
# Clone the repository
git clone https://github.com/kookie12/FlowDrag.git
cd FlowDrag

# Create conda environment
conda env create -f environment_flowdrag.yaml
conda activate flowdrag

```

## 🚀 Usage

### Interactive Editing

Launch the Gradio interface for interactive drag-based editing:

```bash
python flowdrag_ui.py
```

The UI will open in your browser where you can:
1. Upload an image
2. Place handle points (points to drag) and target points (destinations)
3. Adjust deformation parameters
4. Generate edited results

### Mesh Deformation UI

For 3D mesh-guided flow field generation (recommended on local machine for Open3D visualization):

```bash
cd mesh_deformation
python flowdrag_mesh_deform_ui.py
```

This generates `{sample_name}_vector_field.npy` files from input meshes.

## 📊 Evaluation

### VFD-Bench Dataset

Download the VFD-Bench dataset:
- **Access Form**: [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSdYTIpSciFC24xXcKPGgFuTTsWhlTvcxhofVD5guBbbUvcPhg/viewform?usp=dialog)

Place the dataset in `dataset/VFD_Bench_Dataset/`.

### LoRA Training

Train LoRA weights for all samples in the VFD-Bench dataset:

```bash
# Set paths (optional, defaults provided)
export INPUT_DATASET_PATH="dataset/VFD_Bench_Dataset"
export OUTPUT_LORA_PATH="lora_data/VFD_Bench_Dataset"

# Run training
python evaluation/run_lora_training_vfd_bench.py
```

LoRA weights will be saved in `lora_data/VFD_Bench_Dataset/`.

### Run Benchmark

Evaluate FlowDrag on the VFD-Bench dataset:

```bash
python bench_flowdrag.py
```

Results will be saved in the `VFD_Bench_result_flowdrag/` folder, including:
- Edited images
- Concatenated visualizations
- Quantitative metrics


## 🙌🏻 Acknowledgments

Our code is built upon the following excellent projects:
- [GoodDrag](https://github.com/zewei-Zhang/GoodDrag)
- [DragDiffusion](https://github.com/Yujun-Shi/DragDiffusion)
- [DragGAN](https://github.com/XingangPan/DragGAN)

We thank the authors for their great work!


## 📖 Citation

If you find our work useful, please consider citing:

```bibtex
@article{koo2025flowdrag,
  title={Flowdrag: 3d-aware drag-based image editing with mesh-guided deformation vector flow fields},
  author={Koo, Gwanhyeong and Yoon, Sunjae and Lee, Younghwan and Hong, Ji Woo and Yoo, Chang D},
  journal={arXiv preprint arXiv:2507.08285},
  year={2025}
}
```

### Acknowledgement (Funding)
```
This work was partly supported by Institute for Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. 2021-0-01381, Development of Causal AI through Video Understanding and Reinforcement Learning, and Its Applications to Real Environments) and partly supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.2022-0-00184, Development and Study of AI Technologies to Inexpensively Conform to Evolving Policy on Ethics).
``` 
