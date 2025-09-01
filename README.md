# Golden Partition Zone: Rethinking Neural Network Partitioning Under Inversion Threats in Collaborative Inference

This repository provides the official implementation supporting the paper:  
**Golden Partition Zone: Rethinking Neural Network Partitioning Under Inversion Threats in Collaborative Inference**.  

It contains dataset loaders, target model architectures, inversion models, and training scripts for both classifiers and inversion networks. The framework enables systematic exploration of collaborative inference (CI) partitioning strategies under model inversion threats.

---

## Repository Structure

- **`utilis.py` — Datasets & utilities**  
  Includes loaders for **CIFAR-10 (64×64)**, **FaceScrub**, **CelebA**, **ChestX-ray**, **MNIST/EMNIST/KMNIST**, etc.  
  Each dataset class supports train/test/attack splits to simulate collaborative inference scenarios.

- **`target_model_packages.py` — Target models**  
  Provides classification backbones with block-level outputs:  
  - CNN (`Classifier_4`)  
  - ResNet (`rn18`)  
  - VGG (`vgg16_bn`, `vgg19`)  
  - IR-style networks  
  These models expose **intermediate feature maps (`block_idx`)** to evaluate different partitioning strategies.

- **`inversion_model_packages.py` — Inversion models**  
  Implements architectures to reconstruct inputs from intermediate features:  
  - Deconvolutional decoders (`Inversion_4`)  
  - Residual inversion networks
  - **Enhanced inversion capacity** with deeper blocks and **attention mechanisms**  
  - **Normalization & FFT-based entropy enhancement** modules to increase representation entropy and challenge reconstruction.

- **`train_classifier.py` — Training target classifiers**  
  Standard classifier training with optional losses (label smoothing, Focal Loss, contrastive loss, etc.), logging utilities, and accuracy evaluation.

- **`train_inversion.py` — Training inversion models**  
  Reconstructs inputs from partitioned features.  
  Supports evaluation with **MSE**, **PSNR**, **cosine similarity** (optional SSIM).  
  Integrates **normalization** (via `FeatureNormalizer`) and **FFT-based entropy amplification** (via `Entropy_Enhancer_Net`) to study reconstruction robustness.

---

## Getting Started

### Requirements
- Python 3.8+
- PyTorch ≥ 1.9
- torchvision, numpy, scikit-image, matplotlib, tqdm

```bash
pip install -r requirements.txt
```

### Example: Train a Target Classifier
```bash
python train_classifier.py --dataset chest --target_model cnn --epochs 300 --lr 0.01 --batch-size 128
```

### Example: Train an Inversion Model
```bash
python train_inversion.py --dataset mnist --target_model cnn --block_idx 2 --epochs 100 --lr 0.01
```

**Note:** The commands above are **illustrative examples**.  
For **full parameter options**, please check the code directly.  
You should modify **paths, dataset roots, and model architectures inside the `main()` function** according to your environment and experiment design.

---

## Purpose

This repository is intended **for academic research only**.  
It provides a framework for studying **partitioning strategies (“Golden Partition Zone”)** and their vulnerability to **model inversion attacks** in collaborative inference.  
