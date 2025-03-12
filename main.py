from src.dataset import get_DataLoader
from src.train import train_model
from src.evaluation import run_evaluation
from src.test import model_test

def main():
    # 데이터 로더 저장
    train_loader, val_loader, test_dataset = get_DataLoader()
    
    path = 'best_SSD_model.pth'
    
    train_model(train_loader, val_loader, path=path)
    run_evaluation(val_loader, path=path)
    model_test(test_dataset)

if __name__ == '__main__':
    main()