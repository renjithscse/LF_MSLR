# LF-MSLR: Multimodal Fusion for Sign Language Recognition

## 📌 Overview
**LF-MSLR (Late Fusion-Based Multi-modal Sign Language Recognition)** integrates multiple modalities—**RGB video frames, optical flow, and skeleton data**—to improve sign language recognition accuracy. Our model uses keyframe identification and a **Bidirectional LSTM (BiLSTM) network** for classification, achieving state-of-the-art results.

## 🚀 Key Features
- **Multimodal Fusion**: Combines RGB, Optical Flow, and Skeleton Data.
- **Keyframe Extraction**: Focuses on crucial frames to reduce redundancy.
- **Late Fusion Strategy**: Integrates features from different modalities effectively.
- **High Accuracy**: Achieved **95%+ on Include-50** and **92%+ on SMILE-DSGS** datasets.

## 📂 Dataset Download (SMILE-DSGS)
Due to large file size, download **SMILE-DSGS** manually:
```bash
pip install gdown
gdown "https://github.com/neccam/slt/"
```
## 📂 Dataset Download (INCLUDE-50)
Due to large file size, download **Include-50** manually:
```bash
pip install gdown
gdown "https://zenodo.org/records/4010759"
```


## 🛠 Installation
1. **Clone this repository**:
   ```bash
   git clone https://github.com/renjithscse/LF_MSLR.git
   cd LF-MSLR
   ```
   
---

### **3. `requirements.txt`**
```txt
opencv-python
numpy
tensorflow
keras
mediapipe
torch
torchvision
torch-geometric

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
3 Preprocessing

Use `preprocessing.py` to extract frames, compute optical flow, and extract skeleton data from videos.

4 Data Loading

`dataloader.py` defines the `SignLanguageDataset` class, which loads frames, skeletons, and labels for training.

   ```
5. **Run the model**:
   ```bash
   python train.py
   ```

## 📊 Results & Evaluation
| Model           | Dataset        | Accuracy |
|----------------|---------------|----------|
| LF-MSLR        | Include-50     | **95.02%** |
| LF-MSLR        | SMILE-DSGS     | **92.00%** |
| CNN+BiLSTM     | Include-50     | 94.67% |
| CNN+BiLSTM     | SMILE-DSGS     | 89.26% |

## 📜 Citation
If you use our work, please cite:
```
@article{LFMSLR2024,
  author    = {S Renjith et al.},
  title     = {Multimodal Fusion of RGB Video, Optical Flow, and Skeleton Data for Enhanced Sign Language Recognition},
  journal   = {The Visual Computer},
  year      = {2025},
}
```

## 🤝 Contributing
We welcome contributions! Please submit a pull request or open an issue.

## 📧 Contact
For questions, contact renjiths.cse@gmail.com.


