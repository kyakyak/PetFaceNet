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
├── 📂 models/               # Trained models
├── 📂 src/                  # Source code
│   ├── train.py             # Training script
│   ├── detect.py            # Inference script
│   ├── dataset.py           # Dataset handling
│   ├── utils.py             # Helper functions
├── requirements.txt         # Dependencies
├── README.md                # Project documentation
```

---

## 🚀 Installation
To set up the project, follow these steps:

```sh
git clone https://github.com/kyakyak/PetFaceNet.git
cd PetFaceNet
pip install -r requirements.txt
```

---

## 🏅 Training
To train a new model from scratch:

```sh
python src/train.py --epochs 50 --batch_size 16 --dataset_path data/
```

---

## 🔍 Inference
To detect pet faces in an image:

```sh
python src/detect.py --image_path path/to/image.jpg
```

---

## 📊 Results
Sample detection output:

![Detection Example](assets/detection_example.jpg)

---

## 🛠 Dependencies
This project requires the following libraries:

```plaintext
- Python 3.9+
- PyTorch 2.6
- Torchvision
- OpenCV
- Matplotlib
- NumPy
```

---

## 🔗 References
- [The Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/)
- [PyTorch Official Docs](https://pytorch.org/)

---

## 📝 License
This project is licensed under the MIT License.