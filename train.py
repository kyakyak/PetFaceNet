from src.dataset import get_DataLoader
from src.trainer import train_model

def main():
    # 데이터 로더 저장
    train_loader, val_loader, _ = get_DataLoader()
    
    path = 'best_SSD_model.pth'
    
    train_model(train_loader, val_loader, path=path)

if __name__ == '__main__':
    main()