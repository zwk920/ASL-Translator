import cv2
import requests

wsl2_ip = "172.26.119.243"
url = f"http://{wsl2_ip}:5000/video_feed"

cap = cv2.VideoCapture(0)  # 0 for the default webcam, change it if you have multiple webcams

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ret, buffer = cv2.imencode('.jpg', frame)
    img_data = buffer.tobytes()
    response = requests.post(url, data=img_data)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

