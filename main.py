import cv2
import time
import pafy
# pip install youtube-dl


#url = "https://www.youtube.com/watch?v=RZrtf_Hjvsw&ab_channel=SerradoRiodoRastroaoVivoSerradoRiodoRastroaoVivo"
#url = "https://www.youtube.com/watch?v=IhaAnRCX9H0&t=1s&ab_channel=SerradoRiodoRastroaoVivo"
#url = "https://www.youtube.com/watch?v=s2tssDyJLU0"
url = "https://www.youtube.com/watch?v=o2DakCfKx8Y"
video = pafy.new(url)
best = video.getbest()
cv2.namedWindow('ProjLucas')
cv2.CAP_PROP_POS_AVI_RATIO, 2
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
with open('coco.names', 'r') as f:
    class_names = [cname.strip() for cname in f.readlines()]

try:
    # cap = cv2.VideoCapture(0)
    cap_ = cv2.VideoCapture(best.url)
except:
    print('*erro na camera*')

net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)

while True:
    _cap, frame = cap_.read()
    if not _cap:
        break
    start = time.time()
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)
    end = time.time()

    for (class_id, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(class_id) % len(COLORS)]
        label = f'{class_names[class_id[0]]} : {score}'
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    fps_label = f'FPS: {round((1.0 / (end - start)), 2)}'
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    #frame = cv2.resize(frame, (800, 600))
    cv2.imshow('ProjLucas', frame)

    if cv2.waitKey(1) == 27:
        break

cap_.release()
cv2.destroyWindow('ProjLucas')