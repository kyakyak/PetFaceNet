# src/config.py
import os
import torch

# 프로젝트 루트 경로
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 학습 관련 상수
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 50
STEP_SIZE = 5
GAMMA = 0.1
PATIENCE = 5
DELTA = 0.0001