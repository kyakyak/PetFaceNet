import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from src.config import BASE_DIR, DEVICE
from src.SSD import get_model
from src.dataset import transform

def detect_from_image(image_path, path='best_SSD_model.pth'):
    # 모델 로드
    model = get_model()
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, "models", path), map_location=DEVICE))
    model.eval()
    
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).to(DEVICE).unsqueeze(0)  # 배치 차원 추가

    # 모델 예측 수행
    with torch.no_grad():
        prediction = model(image_tensor)

    # 예측 결과 정리
    pred_boxes = prediction[0]['boxes'].cpu().numpy()
    pred_scores = prediction[0]['scores'].cpu().numpy()
    pred_labels = prediction[0]['labels'].cpu().numpy()

    # 이미지 변환 (Tensor → NumPy)
    image_np = np.array(image)

    # 결과 시각화
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(image_np)

    # 신뢰도 0.5 이상인 Bounding Box만 표시
    for bbox, score, label in zip(pred_boxes, pred_scores, pred_labels):
        if score > 0.5:
            xmin, ymin, xmax, ymax = bbox
            label_str = ('cat' if label == 1 else 'dog')

            # 박스 그리기
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

            # 라벨 텍스트 표시
            ax.text(xmin, ymax + 5, f"{label_str}: {score:.2f}", 
                    fontsize=10, color='red', fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax.set_title(f"Detection Result: {os.path.basename(image_path)}")
    ax.axis("off")

    plt.tight_layout()
    plt.show()