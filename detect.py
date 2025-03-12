from src.dataset import get_DataLoader
from src.detector import model_test

def main():
    # 데이터 로더 저장
    _, _, test_dataset = get_DataLoader()
    
    path = 'best_SSD_model.pth'
    
    model_test(test_dataset, path=path)

if __name__ == '__main__':
    main()