import cv2
from src.utils.img import gray3D
import os

video_folder = 'D:/0619_all_newname/train'
files = [file for file in os.listdir(video_folder)]

rgb_folder = os.path.join(video_folder, "rgb")
gray_folder = os.path.join(video_folder, "gray")
os.makedirs(rgb_folder, exist_ok=True)
os.makedirs(gray_folder, exist_ok=True)

step = 20


for n, f in enumerate(files):
    cnt = 0
    cap = cv2.VideoCapture(os.path.join(video_folder, f))
    print("processing video {}".format(n))

    while True:
        ret, frame = cap.read()
        cnt += 1
        if ret:
            if cnt % step == 0:
                cv2.imwrite(os.path.join(rgb_folder,"{}_{}.jpg".format(f[:-4], cnt)), frame)
            gray = gray3D(frame)
            if cnt % step == 0:
                cv2.imwrite(os.path.join(gray_folder,"{}_{}.jpg".format(f[:-4], cnt)), gray)
        else:
            break
