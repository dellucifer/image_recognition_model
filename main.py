import cv2

# img = cv2.imread('keyboard.jpg')
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

classNames = []
classFile = 'coco.names'

with open(classFile,'rt') as f:
  classNames = f.read().rstrip('\n').split('\n')
# print(classNames)

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightPath,configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=0.5)
    print(classIds,bbox)
    pt1 = (bbox[0][0],bbox[0][1])
    pt2 = (bbox[0][2],bbox[0][3])
    # print(pt1)

    if len(classIds) != 0:
        for classID, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox.flatten()):
            cv2.rectangle(img,pt1,pt2,color=(0,255,0),thickness=2,)
            cv2.putText(img,classNames[classID-1],(pt1[0]+10,pt1[1]+30),cv2.FONT_ITALIC,1,(255,0,0),2)

    cv2.imshow("output", img)
    cv2.waitKey(1)
