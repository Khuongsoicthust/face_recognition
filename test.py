# USAGE
# python3.5 test.py --detector face_detection_model \
# 	--embedding-model openface_nn4.small2.v1.t7 \
# 	--recognizer output/recognizer.pickle \
# 	--le output/le.pickle

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import random
import datetime
from tracker.centroidtracker import CentroidTracker

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-cam", "--camera", required=False, default = 0,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = cv2.VideoCapture('/home/ntk/Desktop/video_1.mp4')
# vs = VideoStream(args["camera"]).start()
# vs = VideoStream(src=0).start()
time.sleep(2.0)
ct = CentroidTracker()
writer = None

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

# start the FPS throughput estimator
fps = FPS().start()

# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream
	(grabbed, frame) = vs.read()
	if not grabbed:
		break
	# frame = vs.read()

	# resize the frame to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	frame = imutils.resize(frame, width=1000)
	(h, w) = frame.shape[:2]

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (1000, 1000)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()
	rects = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > 0.4:
			# compute the (x, y)-coordinates of the bounding box for
			# the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			rects.append(box.astype("int"))

			# extract the face ROI
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			# construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face
			# faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
			# 	(96, 96), (0, 0, 0), swapRB=True, crop=False)
			# embedder.setInput(faceBlob)
			# vec = embedder.forward()

			# # perform classification to recognize the face
			# preds = recognizer.predict_proba(vec)[0]
			# # print(preds)
			# j = np.argmax(preds)
			# # print(preds[j])
			# if preds[j] < 0.9:
				
			# 	cv2.rectangle(frame, (startX, startY), (endX, endY),
			# 	(0, 0, 255), 2)
				
			# else:
	
			# 	proba = preds[j]
			# 	name = le.classes_[j]

			# draw the bounding box of the face along with the
			# associated probability
				# text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
				# cv2.putText(frame, text, (startX, y),
				# 	cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

			# construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face

			# draw the bounding box of the face along with the
			# associated probability

	if writer is None:
	# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter("output/test.avi", fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

	# some information on processing single frame

	# write the output frame to disk
	writer.write(frame)
		
	# loop over the tracked objects
	# objects = ct.update(rects)
	# for (objectID, centroid) in objects.items():
	# 	# draw both the ID of the object and the centroid of the
	# 	# object on the output frame
	# 	text = "Person: {}".format(objectID)

	# 	cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
	# 		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		# cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	# show the output frame
		
	# update the FPS counter
	fps.update()

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# # stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
