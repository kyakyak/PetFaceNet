import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.config import BASE_DIR, DEVICE
from src.SSD import get_model

def model_test(test_dataset, path='best_SSD_model.pth'):
    # 모델 로드
    model = get_model()
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, "models", path)))
    model.eval()

    # 랜덤한 테스트 이미지 인덱스 선택 (5개)
    num_images = 5
    random_indices = random.sample(range(len(test_dataset)), num_images)

    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 4, 6))

    for idx, img_index in enumerate(random_indices):
        # 테스트 이미지 가져오기
        image, img_name = test_dataset[img_index]
        image = image.to(DEVICE).unsqueeze(0)  # 배치 차원 추가

        # 모델 예측 수행
        with torch.no_grad():
            prediction = model(image)

        # 예측 결과 정리
        pred_boxes = prediction[0]['boxes'].cpu().numpy()
        pred_scores = prediction[0]['scores'].cpu().numpy()
        pred_labels = prediction[0]['labels'].cpu().numpy()

        # 이미지 변환 (Tensor → NumPy)
        image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image_np = np.clip(image_np, 0, 1)

        # 결과 시각화
        ax = axes[idx] if num_images > 1 else axes
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
                ax.text(xmin, ymax + 5, f"Class {label_str}: {score:.2f}", 
                        fontsize=10, color='red', fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        ax.set_title(f"Image: {img_name}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()