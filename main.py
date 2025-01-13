import cv2 
import numpy as np


pen_template = cv2.imread("pen.jpg", 0)
notebook_template = cv2.imread("notebook.jpg", 0)

if pen_template is None or notebook_template is None:
    print("Error loading images")
    exit()


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

   
    pen_res = cv2.matchTemplate(gray_frame, pen_template, cv2.TM_CCOEFF_NORMED)
    notebook_res = cv2.matchTemplate(gray_frame, notebook_template, cv2.TM_CCOEFF_NORMED)

    
    pen_thresh = 0.7
    notebook_thresh = 0.7

    pen_loc = np.where(pen_res >= pen_thresh)
    notebook_loc = np.where(notebook_res >= notebook_thresh)

    
    if len(pen_loc[0]) > 0:
        print("Pen detected!")

    if len(notebook_loc[0]) > 0:
        print("Notebook detected!")

    
    cv2.imshow("Object Detection", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
