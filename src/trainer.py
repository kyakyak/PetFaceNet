import os
import torch
import torch.optim as optim
from tqdm import tqdm
from src.config import BASE_DIR, DEVICE, LEARNING_RATE, NUM_EPOCHS, STEP_SIZE, GAMMA, PATIENCE, DELTA
from src.SSD import get_model

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0001, verbose=False, path='best_model.pth'):
        """
        patience: 개선이 없을 경우 종료까지 기다릴 에폭 수
        delta: 최소 개선 값
        verbose: 모델 저장 여부를 출력할지 여부
        path: 모델 저장 경로
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = os.path.join(BASE_DIR, "models", path)
        self.counter = 0
        self.best_score = -float("inf")  # 초기값을 -무한대로 설정
        self.early_stop = False

    def __call__(self, score, model):
        """검증 점수를 기준으로 Early Stopping 수행"""
        if self.patience == 0:
            self.save_checkpoint(model)  # 모델 저장
            return

        if score > self.best_score + self.delta:  # 스코어가 개선되었으면
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0  # 카운터 초기화
        else:  # 개선되지 않았으면 카운터 증가
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        """Validation 점수가 개선될 때 모델 저장"""
        if self.verbose:
            print(f"Best score improved to {self.best_score:.4f}. Saving model...")
        torch.save(model.state_dict(), self.path)

def evaluate_model(model, val_loader):
    model.eval()
    max_scores = []

    with torch.no_grad():
        for images, _ in val_loader:
            images = [img.to(DEVICE) for img in images]
            pre = model(images)

            for p in pre:
                scores = p['scores'].cpu().numpy()
                if len(scores) > 0:
                    max_scores.append(float(max(scores)))
                else:
                    max_scores.append(0.0)

    return float(sum(max_scores)) / len(max_scores) if len(max_scores) > 0 else 0.0

def train_model(train_loader, val_loader, path='best_SSD_model.pth'):
    model = get_model()
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA) 
    early_stopping = EarlyStopping(patience=PATIENCE, delta=DELTA, verbose=True, path=path)

    train_losses = []
    val_scores = []
    
    # 학습 시작
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0
        
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images = [img.to(DEVICE) for img in images]  # 이미지 변환
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]  # 바운딩 박스 및 라벨을 GPU로 이동

            optimizer.zero_grad()
            loss_dict = model(images, targets)  # SSD 모델 학습
            loss = sum(loss for loss in loss_dict.values())  # 총 손실 계산
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        val_score = evaluate_model(model, val_loader)
        val_scores.append(val_score)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f}, Val Score: {val_score:.4f}")
        
        early_stopping(val_score, model)  # 검증 점수를 기준으로 조기 종료 확인
        if early_stopping.early_stop:  
            print("Early stopping activated")
            break  # 학습 중단