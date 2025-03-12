import argparse
from src.detector import detect_from_image

def main():
    parser = argparse.ArgumentParser(description="Detect pet faces in an image.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    
    path = "best_SSD_model.pth"

    args = parser.parse_args()
    
    detect_from_image(args.image_path, path=path)

if __name__ == '__main__':
    main()