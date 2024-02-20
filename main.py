import os
import cv2
from ultralytics import YOLO
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QFileDialog


class VideoAnalysis(QWidget):
    def __init__(self):
        super().__init__()
        self.ui()

    def ui(self):
        self.setWindowTitle('Morris Water Maze Analyst')
        self.setGeometry(100, 100, 400, 400)

        self.videoPath = None

        self.selectButton = QPushButton('Select Video', self)
        self.selectButton.setGeometry(100, 50, 150, 30)
        self.selectButton.clicked.connect(self.select_video)

        self.analysisButton = QPushButton('Analyse Video', self)
        self.analysisButton.setGeometry(100, 100, 150, 30)
        self.analysisButton.clicked.connect(self.analyze_video)
        self.analysisButton.setEnabled(False)

        self.windowInfo = QLabel(self)
        self.windowInfo.setGeometry(100, 200, 400, 80)

    def select_video(self):

        self.videoPath, _ = QFileDialog.getOpenFileName(self, 'Select Video')

        if self.videoPath:
            self.windowInfo.setText(f'Selected video:\n{self.videoPath}')
            self.analysisButton.setEnabled(True)

    def analyze_video(self):
        if self.videoPath:

            video = cv2.VideoCapture(self.videoPath)
            ret, frame = video.read()
            height, width, color_dept = frame.shape

            output_path = "./rat-detection/result.mp4"
            video_result = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), int(video.get(cv2.CAP_PROP_FPS)), (width, height))

            model_path = "./rat-detection/runs/detect/train/weights/last.pt"
            model = YOLO(model_path)
            threshold = 0.5

            platform_location = True
            image_path = r'./rat-detection/frame.jpg'
            image_result = cv2.imread(image_path)

            rats_path = []
            red = 255
            green = 0

            while ret:

                results = model(frame)[0]


                for result in results.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = result

                    if score > threshold:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                        cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1 ,(0, 255, 0), 2, cv2.LINE_AA)

                        if results.names[int(class_id)].upper().__eq__("RAT"):
                            rat_path_coordinates = (int((int(x1) + int(x2)) // 2), int((int(y1) + int(y2)) // 2))
                            cv2.circle(image_result, rat_path_coordinates, 2, (0, green, red), 3)
                            red -= 0.25
                            green += 0.25
                            rats_path.append(rat_path_coordinates)

                        if results.names[int(class_id)].upper().__eq__("PLATFORM") and platform_location:
                            platform_center_coordinates = (int((int(x1) + int(x2)) // 2), int((int(y1) + int(y2)) // 2))
                            radius = ((int(x2) - int(x1)) + (int(y2) - int(y1))) // 4
                            color = (255, 0, 0)
                            thickness = 2
                            image_result = cv2.circle(image_result, platform_center_coordinates, radius, color, thickness)
                            platform_location = False

                video_result.write(frame)
                ret, frame = video.read()

            cv2.imwrite('image_result.jpg', image_result)
            video.release()
            video_result.release()
            cv2.destroyAllWindows()
            self.windowInfo.setText(f'Video analysis was successfully!\nVideo saved as:\n {output_path}')


if __name__ == '__main__':
    app = QApplication([])
    window = VideoAnalysis()
    window.show()
    app.exec_()
