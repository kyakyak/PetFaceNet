# PetFaceNet 🐶🐱

AI-powered pet face detection and classification. Detects and classifies dog and cat faces using deep learning.

---

## 📌 Features
- Detects dog and cat faces using object detection (SSD).
- Classifies detected faces into "Dog" or "Cat".
- Supports training custom models on The Oxford-IIIT Pet Dataset.
- Exports trained models in `.pth` format.

---

## 📚 Project Structure
```
📂 PetFaceNet
├── 📂 data/                 # Dataset storage
│   ├── 📂 images/           # Raw images
│   ├── 📂 annotations/      # Annotation files
├── 📂 models/               # Trained models (Saved models in .pth format)
├── 📂 notebooks/            # Jupyter Notebook containing full training workflow
│   ├── notebook.ipynb       # Main notebook for training & evaluation
├── 📂 src/                  # Source code
│   ├── config.py            # Configuration settings (batch size, paths, etc.)
│   ├── dataset.py           # Dataset handling & preprocessing
│   ├── detector.py          # Model inference and face detection
│   ├── evaluation.py        # Model evaluation metrics (mAP)
│   ├── SSD.py               # SSD model architecture
│   ├── tester.py            # Model image detect test
│   ├── trainer.py           # Training loop and logic
├── .gitignore               # Ignore unnecessary files
├── .gitattributes           # Git LFS tracking for large files
├── README.md                # Project documentation
├── requirements.txt         # Required dependencies
├── main.py                  # Main script for training and evaluation
├── train.py                 # Training script 
├── detect.py                # Inference script to detect pet faces
├── Example.png              # Example image
├── LICENSE                  # MIT License
```

---

## 🚀 Installation
To set up the project, follow these steps:

```sh
git clone https://github.com/kyakyak/PetFaceNet.git
cd PetFaceNet
pip install -r requirements.txt
```

### ❗ If the model is not loading
This project uses **Git LFS (Large File Storage)** to store large files such as trained models.

If the model file is missing or not downloaded properly, try the following:

1. Install Git LFS (only required once):
   ```sh
   git lfs install
   ```

2. Pull the model files manually:
   ```sh
   git lfs pull
   ```

This should ensure that all model files are correctly downloaded.

---

## 🏅 Training
To train a new model from scratch:

```sh
python train.py
```
The batch size, number of epochs, learning rate, and other settings can be adjusted in the src/config.py file. Modify the values as needed before running train.py.

---

## 🔍 Inference
To detect pet faces in an image:

```sh
python detect.py --image_path path/to/image.jpg
```

---

## 📊 Results
Sample detection output:

![Detection Example](./Example.png)

---

## 🛠 Dependencies
This project requires the following libraries:

```plaintext
- Python 3.9+
- PyTorch 2.6
- Torchvision
- Matplotlib
- NumPy
```

---

## 🔗 References
- [The Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/)
- [PyTorch Official Docs](https://pytorch.org/)

---

## 📝 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.