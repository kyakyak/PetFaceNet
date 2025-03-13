import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from src.config import BASE_DIR, DEVICE
from src.SSD import get_model
from src.dataset import transform  # 이미지 전처리 변환 함수

def detect_from_image(image_path, path='best_SSD_model.pth'):
    # 모델 로드
    model = get_model()
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, "models", path), map_location=DEVICE))
    model.eval()
    
    # 원본 이미지 로드
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size  # 원본 이미지 크기 저장

    # 이미지 전처리 (리사이징 등 적용)
    transformed_image = transform(image)
    image_tensor = transformed_image.to(DEVICE).unsqueeze(0)  # 배치 차원 추가

    # 모델 예측 수행
    with torch.no_grad():
        prediction = model(image_tensor)

    # 예측 결과 정리
    pred_boxes = prediction[0]['boxes'].cpu().numpy()
    pred_scores = prediction[0]['scores'].cpu().numpy()
    pred_labels = prediction[0]['labels'].cpu().numpy()

    # 변환된 이미지 크기 가져오기 (모델 입력 크기)
    _, transformed_h, transformed_w = transformed_image.shape  # (C, H, W)

    # 바운딩 박스를 원본 이미지 크기에 맞게 스케일 변환
    scale_x = orig_w / transformed_w
    scale_y = orig_h / transformed_h
    pred_boxes[:, [0, 2]] *= scale_x  # x 좌표 조정 (xmin, xmax)
    pred_boxes[:, [1, 3]] *= scale_y  # y 좌표 조정 (ymin, ymax)

    # 결과 시각화
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(np.array(image))  # 원본 이미지 시각화

    # 신뢰도 0.5 이상인 Bounding Box만 표시
    for bbox, score, label in zip(pred_boxes, pred_scores, pred_labels):
        if score > 0.5:
            xmin, ymin, xmax, ymax = bbox
            label_str = 'cat' if label == 1 else 'dog'

            # 박스 그리기 (원본 이미지 좌표 기준)
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