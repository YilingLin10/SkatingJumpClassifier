import cv2
import os 

type = "E"
fileName = "e09"
cap = cv2.VideoCapture("20220526調整動作擷取/{}/{}.mp4".format(type, fileName))
success,image = cap.read()
i = 0

os.makedirs("{}/{}/".format(type, fileName), exist_ok=False)

if not success:
    print("failed to read file: {}".format("20220526調整動作擷取/{}/{}.mp4".format(type, fileName)))

while(success):
    cv2.imwrite("{}/{}/".format(type, fileName) + str(i) + '.jpg', image)
    success,image = cap.read()
    i+=1

cap.release()
cv2.destroyAllWindows()
