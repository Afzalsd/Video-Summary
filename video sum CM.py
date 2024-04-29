import cv2
import numpy as np


#change input video path
video = cv2.VideoCapture('_xMr-HKMfVA.mp4')
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
threshold = 1500000.

#change output video path
writer = cv2.VideoWriter('_xMr-HKMfVA copy.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25, (width, height))
ret, frame1 = video.read()
prev_frame = frame1
last_frame_num = video.get(cv2.CAP_PROP_FRAME_COUNT)
a = 0
b = 0
c = 0
co = 0

protopath = 'MobileNetSSD_deploy.prototxt'
modelpath = 'MobileNetSSD_deploy.caffemodel'

detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Ground truth labels (assuming all frames have bicycles)
ground_truth_labels = np.ones((int(last_frame_num),))

# Initialize variables for confusion matrix
true_positives = 0
false_positives = 0
false_negatives = 0
true_negatives = 0

while True:
    ret, frame = video.read()
    if co + 1 == last_frame_num:
        break

    (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
    detector.setInput(blob)
    person_detections = detector.forward()
    co = co + 1

    for i in np.arange(0, person_detections.shape[2]):
        confidence = person_detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(person_detections[0, 0, i, 1])

            # change the threshold value according to your requirement1710
            if CLASSES[idx] != "person":
                continue

            person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = person_box.astype("int")

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

            if (np.sum(cv2.absdiff(frame, prev_frame)) > threshold):
                writer.write(frame)
                prev_frame = frame
                cv2.imshow("Application", frame)
                a += 1
                ground_truth_labels = a
            else:
                prev_frame = frame
                b += 1

            # Calculate IoU (Intersection over Union) for evaluating detection accuracy
            intersection = np.maximum(0, np.minimum(endX, person_box[2]) - np.maximum(startX, person_box[0])) * \
                           np.maximum(0, np.minimum(endY, person_box[3]) - np.maximum(startY, person_box[1]))
            union = (endX - startX) * (endY - startY) + (person_box[2] - person_box[0]) * (person_box[3] - person_box[1]) - intersection
            iou = intersection / union

            # Update confusion matrix based on IoU
            if iou > 0.5:
                true_positives += 1
            else:
                false_positives += 1

    c += 1
    key = cv2.waitKey(1)

# Calculate false negatives
false_negatives = (a+b) - true_positives
true_negatives = b - false_positives

# Calculate precision, recall, and F1 score
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1_score = 2 * (precision * recall) / (precision + recall)
accuracy= ( true_positives + true_negatives) / (true_positives+ false_positives + true_negatives + false_negatives)

print("Total frames: ", c)
print("Unique frames: ", a)
print("Common frames: ", b)
print("True Positives: ", true_positives)
print("False Positives: ", false_positives)
print("False Negatives: ", false_negatives)
print("True Negatives: ", true_negatives)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1_score)
print("accuracy: ", accuracy)

video.release()
writer.release()
cv2.destroyAllWindows()
