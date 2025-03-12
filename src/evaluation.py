import os
import torch
import numpy as np
from src.config import BASE_DIR, DEVICE
from src.SSD import get_model

def calculate_iou(box1, box2):
    """
    IoU (Intersection over Union) 계산
    box1, box2: (xmin, ymin, xmax, ymax)
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = interArea / float(box1Area + box2Area - interArea)
    return iou

def evaluate_map(model, data_loader, iou_threshold=0.5):
    """
    SSD 모델의 Mean Average Precision (mAP) 평가 함수
    - model: 평가할 모델
    - data_loader: 검증 데이터 로더
    - iou_threshold: TP를 결정하는 IoU 기준 (일반적으로 0.5)
    """
    model.eval()
    all_detections = []  # 예측된 바운딩 박스
    all_ground_truths = []  # 실제 바운딩 박스

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            predictions = model(images)

            for target, prediction in zip(targets, predictions):
                gt_boxes = target["boxes"].cpu().numpy()
                gt_labels = target["labels"].cpu().numpy()

                pred_boxes = prediction["boxes"].cpu().numpy()
                pred_labels = prediction["labels"].cpu().numpy()
                pred_scores = prediction["scores"].cpu().numpy()

                all_ground_truths.append((gt_boxes, gt_labels))
                all_detections.append((pred_boxes, pred_labels, pred_scores))

    return compute_map(all_ground_truths, all_detections, iou_threshold)

def compute_map(ground_truths, detections, iou_threshold):
    """
    Mean Average Precision (mAP) 계산 함수
    - ground_truths: 실제 바운딩 박스 리스트
    - detections: 예측된 바운딩 박스 리스트
    - iou_threshold: TP 판단 기준
    """
    aps = []

    # 클래스별 AP 계산
    for class_id in range(1, 3):  # SSD에서는 보통 1부터 시작 (고양이: 1, 개: 2)
        all_gt_boxes = []
        all_pred_boxes = []
        all_scores = []
        num_gt_boxes = 0

        for gt, pred in zip(ground_truths, detections):
            gt_boxes, gt_labels = gt
            pred_boxes, pred_labels, pred_scores = pred

            gt_mask = gt_labels == class_id
            pred_mask = pred_labels == class_id

            gt_boxes = gt_boxes[gt_mask]
            pred_boxes = pred_boxes[pred_mask]
            pred_scores = pred_scores[pred_mask]

            all_gt_boxes.extend(gt_boxes)
            all_pred_boxes.extend(pred_boxes)
            all_scores.extend(pred_scores)
            num_gt_boxes += len(gt_boxes)

        # Precision-Recall Curve 계산
        precision, recall = compute_precision_recall(all_gt_boxes, all_pred_boxes, all_scores, num_gt_boxes, iou_threshold)
        ap = compute_ap(precision, recall)
        aps.append(ap)

    return np.mean(aps)  # 모든 클래스의 AP 평균값

def compute_precision_recall(gt_boxes, pred_boxes, scores, num_gt_boxes, iou_threshold):
    """
    Precision-Recall Curve 계산
    """
    if len(pred_boxes) == 0:
        return [0], [0]

    sorted_indices = np.argsort(-np.array(scores))
    pred_boxes = np.array(pred_boxes)[sorted_indices]

    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))

    matched = set()
    for i, pred_box in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1
        for j, gt_box in enumerate(gt_boxes):
            if j in matched:
                continue
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou > iou_threshold:
            tp[i] = 1
            matched.add(best_gt_idx)
        else:
            fp[i] = 1

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    recall = tp / (num_gt_boxes + 1e-6)
    precision = tp / (tp + fp + 1e-6)

    return precision, recall

def compute_ap(precision, recall):
    """
    Average Precision (AP) 계산
    """
    recall = np.concatenate(([0], recall, [1]))
    precision = np.concatenate(([0], precision, [0]))

    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap

def run_evaluation(loader, path='best_SSD_model.pth'):
    # 모델 로드
    model = get_model()
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, "models", path)))

    map_05 = evaluate_map(model, loader, iou_threshold=0.5)
    map_075 = evaluate_map(model, loader, iou_threshold=0.75)
    map_09 = evaluate_map(model, loader, iou_threshold=0.9)

    print(f"mAP 0.50: {map_05:.4f}")
    print(f"mAP 0.75: {map_075:.4f}")
    print(f"mAP 0.90: {map_09:.4f}")