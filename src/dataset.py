import os
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from src.config import BASE_DIR, DEVICE, BATCH_SIZE

# 클래스 매핑
CLASS_MAPPING = {"cat": 1, "dog": 2}

# 데이터 변환 설정
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor()
])

class PetDataset(Dataset):
    def __init__(self, data_dir, txt_file, transform=None, target_size=(300, 300)):
        """
        Oxford-IIIT Pet Dataset을 위한 PyTorch Dataset 클래스 (Bounding Box 포함)
        """
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "images")
        self.anno_dir = os.path.join(data_dir, "annotations", "xmls")
        self.transform = transform
        self.target_size = target_size  # (300, 300) 고정
        self.image_list = []

        # 유효한 이미지 & XML만 리스트에 추가
        with open(txt_file, "r") as f:
            for line in f.readlines():
                img_name = line.strip().split()[0] + ".jpg"
                xml_name = img_name.replace(".jpg", ".xml")

                img_path = os.path.join(self.image_dir, img_name)
                xml_path = os.path.join(self.anno_dir, xml_name)

                # 이미지와 XML 파일이 모두 존재하는 경우만 추가
                if os.path.exists(img_path) and os.path.exists(xml_path):
                    self.image_list.append(img_name)

        print(f"학습 데이터 개수: {len(self.image_list)}개 (필터링 완료)")

    def parse_annotation(self, xml_file, original_size):
        """
        XML 어노테이션 파일을 파싱하여 Bounding Box 및 클래스 라벨 정보 반환 (리사이징 반영)
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()
        width_orig, height_orig = original_size  # 원본 이미지 크기

        width_scale = self.target_size[0] / width_orig
        height_scale = self.target_size[1] / height_orig

        objects = []
        labels = []

        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text) * width_scale
            ymin = int(bbox.find("ymin").text) * height_scale
            xmax = int(bbox.find("xmax").text) * width_scale
            ymax = int(bbox.find("ymax").text) * height_scale
            objects.append([xmin, ymin, xmax, ymax])

            # 객체의 클래스명 가져오기 (cat or dog)
            class_name = obj.find("name").text.lower()
            labels.append(CLASS_MAPPING.get(class_name, 1))  # 기본값 cat(1)

        return objects, labels

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # 이미지 로드
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"이미지 파일이 존재하지 않습니다: {img_path}")

        image = Image.open(img_path).convert("RGB")
        original_size = image.size  # (width, height)

        # 어노테이션 로드
        xml_name = img_name.replace(".jpg", ".xml")
        xml_path = os.path.join(self.anno_dir, xml_name)

        if os.path.exists(xml_path):
            bboxes, class_labels = self.parse_annotation(xml_path, original_size)
        else:
            bboxes, class_labels = [], []

        # SSD 학습을 위한 `targets` 변환 (배치 데이터 구조 수정)
        boxes = torch.tensor(bboxes, dtype=torch.float32).to(DEVICE) if len(bboxes) > 0 else torch.zeros((0, 4), dtype=torch.float32).to(DEVICE)
        labels = torch.tensor(class_labels, dtype=torch.int64).to(DEVICE) if len(class_labels) > 0 else torch.zeros((0,), dtype=torch.int64).to(DEVICE)

        targets = {"boxes": boxes, "labels": labels}

        # 이미지 변환 적용
        if self.transform:
            image = self.transform(image)

        return image, targets

class TestPetDataset(Dataset):
    def __init__(self, data_dir, txt_file, transform=None):
        """
        Oxford-IIIT Pet Dataset을 위한 테스트용 PyTorch Dataset 클래스 (Bounding Box 없음)
        :param data_dir: 데이터가 저장된 최상위 폴더 경로
        :param txt_file: test.txt 파일 경로
        :param transform: 이미지 변환 (Torchvision Transform)
        """
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "images")
        self.transform = transform
        self.image_list = []

        # 존재하는 이미지 파일만 리스트에 추가
        with open(txt_file, "r") as f:
            for line in f.readlines():
                img_name = line.strip().split()[0] + ".jpg"
                img_path = os.path.join(self.image_dir, img_name)

                if os.path.exists(img_path):
                    self.image_list.append(img_name)

        print(f"테스트 데이터 개수: {len(self.image_list)}개 (필터링 완료)")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # 이미지 로드
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"이미지 파일이 존재하지 않습니다: {img_path}")

        image = Image.open(img_path).convert("RGB")

        # 변환 적용
        if self.transform:
            image = self.transform(image)

        return image, img_name  # 테스트셋에서는 파일명도 함께 반환

def collate_fn(batch):
    """
    DataLoader에서 SSD 모델이 요구하는 `targets` 형태를 유지하기 위한 함수.
    - `images`: 텐서 리스트 (batch 형태 유지)
    - `targets`: 리스트 (batch 단위로 묶이지 않도록)
    """
    images, targets = zip(*batch)  # 이미지와 타겟을 분리

    images = torch.stack(images, dim=0)  # 이미지들은 배치 단위로 스택 쌓기
    return images, list(targets)  # targets는 리스트 형태 유지 (batch 단위로 묶이지 않게)

def get_DataLoader():
    # 데이터 경로
    data_root = os.path.join(BASE_DIR, "data")

    # 데이터셋 로드
    # 전체 trainval dataset 로드
    trainval_dataset = PetDataset(data_root, os.path.join(data_root, "annotations", "trainval.txt"), transform=transform)

    # Train/Validation Split (80:20)
    train_size = int(0.8 * len(trainval_dataset))
    val_size = len(trainval_dataset) - train_size
    train_dataset, val_dataset = random_split(trainval_dataset, [train_size, val_size])

    test_dataset = TestPetDataset(data_root, os.path.join(data_root, "annotations", "test.txt"), transform=transform)
    
    # 데이터로더 생성   
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_dataset