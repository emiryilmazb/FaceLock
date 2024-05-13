from ultralytics import YOLO

model = YOLO('models/yolov8n.pt')

def main():
    model.train(data='Dataset/SplitData/data.yaml', epochs=3)


if __name__ == '__main__':
    main()
