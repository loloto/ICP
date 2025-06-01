# ICP: Immediate Compensation Pruning for Mid‑to‑High Sparsity

Official implementation for the paper **“ICP: Immediate Compensation Pruning for Mid‑to‑High Sparsity.”**
([arXiv link ― coming soon](#))

> *Xin Luo, Xueming Fu, Zihang Jiang, S. Kevin Zhou*
> University of Science and Technology of China, MIRACLE Center & CAS ICT

---

## 📝 Overview

ICP strikes a middle ground between costly iterative pruning and purely one‑shot pruning by **compensating inter‑block errors on‑the‑fly**.
Key highlights:

| Feature                           | Description                                                                                               |
| --------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **Block‑wise Compensate Pruning** | Alternating *prune → compensate* in a sliding window of two Transformer blocks, preventing error cascade. |
| **Sparsity Rearrangement**        | Re‑allocates sparsity both *between* blocks (α) and *within* a block (β) to minimise uncompensated error. |
| **Language & Vision support**     | Works out‑of‑the‑box on OPT/Llama (NLP) **and** SAM (CV) backbones.                                       |
| **Memory‑friendly**               | Only one block resides on GPU at any time; runs on a single RTX 3090 (24 GB).                             |

Extensive experiments show superior perplexity / accuracy / IoU versus SparseGPT, Wanda, and magnitude pruning across sparsity **50–90 %** and mixed 2:4 / 4:8 + int3/int4 compression.

---

## 📂 Repository Structure

```
├── icp_sam.py        # ICP for Segment Anything Model
├── icp_opt.py        # ICP for OPT language models
├── icp_llama.py      # ICP for Llama / Llama‑2 models
├── bash/             # One‑click scripts with tuned hyper‑parameters
├── segment_anything/ # Minimal SAM dependency (copied from Meta)
├── data_utils.py | feature_utils.py | ...
└── README.md         # <— you are here
```

*All other files are auxiliary helpers (loading, evaluation, quantisation, etc.).*

---

## ⚙️ Requirements

* Python ≥ 3.9
* PyTorch ≥ 2.1 (CUDA 11.8)
* transformers ≥ 4.41
* accelerate, einops, datasets, tqdm, …

Create an environment (example):

```bash
conda create -n icp python=3.10 pytorch=2.2.2 cuda=11.8 -c pytorch -c nvidia
conda activate icp
pip install -r requirements.txt  # list provided
```

---

## 📥 Checkpoints & Datasets

* **OPT / Llama models** are auto‑downloaded from Hugging Face on first run.
* **SAM models** ― download the Meta SAM checkpoints and COCO 2017 + SA‑1B splits:
  `sa_000001.tar`, `sa_000003.tar` → extract to `sa_000001/`, `sa_000003/` and update the paths below.

---

## 🚀 Quick Start

```bash
# Language models
bash icp_opt.sh      # prune OPT‑125M/1.3B/2.7B/6.7B
bash icp_llama.sh    # prune Llama‑2‑7B

# Vision model
bash icp_sam.sh      # prune SAM‑B/L/H
```

Each script exposes CLI flags for sparsity (`--sparsity`), block window (`--alpha`), intra‑block β, calibration size, epochs, quantisation bits, etc.

---

## ▶️ **Run Instructions (fill in later)**

> *TODO: Luo Xin – please paste the detailed step‑by‑step commands here (training, evaluation, reproducing tables 1‑8).*
> The section is intentionally left blank for your upcoming notes.

---

## 📊 Results

---

## ✏️ Citation

```bibtex
@inproceedings{luo2025icp,
  title     = {ICP: Immediate Compensation Pruning for Mid-to-High Sparsity},
  author    = {Xin Luo and Xueming Fu and Zihang Jiang and S. Kevin Zhou},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025}
}
```

---

## 🛡 License

This repository is released under the **CC BY‑NC 4.0** license — free for research & non‑commercial use.
See the [LICENSE](LICENSE) file for the full text.

---

## 🤝 Acknowledgements

Parts of the code are adapted from [SparseGPT](https://github.com/IST-DASLab/sparsegpt), [Wanda](https://github.com/locuslab/wanda) and Meta’s [segment-anything](https://github.com/facebookresearch/segment-anything). We thank the open‑source community!
