import cv2
from ultralytics import YOLO # object tracking
import mediapipe as mp # pose estimation

from transformers import pipeline # depth detection
from PIL import Image
import numpy as np

def point_in_box(point, p1, p2):
    # comparing each coordinate
    for i in range(0, len(point)):
        if point[i] < p1[i] or point[i] > p2[i]:
            return False
    return True

def get_colours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
    (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

inp = input('0: Camera (real-time), 1: Video Path\n')
if int(inp) == 0:
    videoCap = cv2.VideoCapture(0)
else:
    v_path = input("Path: ")
    videoCap = cv2.VideoCapture(v_path)

yolo = YOLO('yolov8s.pt')

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

pipe = pipeline(task = "depth-estimation", model="LiheYoung/depth-anything-small-hf")


initial_px_height = {}
relative_distance = {}
left_foot_pos = {}
right_foot_pos = {}
left_hip_pos = {}
right_hip_pos = {}

while videoCap.isOpened():
    ret, frame = videoCap.read()
    if not ret:
        continue
    results = yolo.track(frame, stream=True)
    # Object detection and visualization code

    #depth detection
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    depth = pipe(image)["depth"]

    pose_results = None
    mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    print(pose_results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x, pose_results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].z)

    people_boxes = []
    for result in results:
        classes_names = result.names
        for box in result.boxes:
            if box.conf[0] > 0.4:
                if classes_names[int(box.cls[0])] == "person":
                    people_boxes.append(box)

    for i in range(0, len(people_boxes)):
        box = people_boxes[i]
        if box.conf[0] > 0.4:
            [x1, y1, x2, y2] = box.xyxy[0] # floats
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # converting to ints for pixel drawing

            cls = int(box.cls[0])
            class_name = classes_names[cls]
            colour = get_colours(cls)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

            roi = frame[y1:y2, x1:x2].copy()
            pose_results = pose.process(roi)
            mp_drawing.draw_landmarks(roi, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            frame[y1:y2, x1:x2] = roi

            ## CALCULATING LANDMARK POSITIONS
            height, width, _ = frame.shape

            try:
                left_foot = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
                left_foot_pos[i] = (int(left_foot.x * width), int(left_foot.y * height))
            except:
                left_foot_pos[i] = (-1, -1)

            try:
                right_foot = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
                right_foot_pos[i] = (int(right_foot.x * width), int(right_foot.y * height))
            except:
                right_foot_pos[i] = (-1, -1)

            try:
                left_hip = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                left_hip_pos[i] = (int(left_hip.x * width), int(left_hip.y * height))
            except:
                left_hip_pos[i] = (-1, -1)

            try:
                right_hip = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
                right_hip_pos[i] = (int(right_hip.x * width), int(right_hip.y * height))
            except:
                right_hip_pos[i] = (-1, -1)
    
    for i in range(0, len(people_boxes)): # runs after all poses have been calculated
        box = people_boxes[i]
        if box.conf[0] > 0.4:
            # distance calculation based on height
            if i not in initial_px_height or initial_px_height[i] <= 0:
                initial_px_height[i] = y2 - y1
            else:
                height = y2 - y1
                relative_distance[i] = initial_px_height[i] / height

            # distance calculation based on foot position

            # relative distance calculation based on depth estimation
            ## determine that key pose landmarks are not obstructed by other people:
            left_hip_valid = True
            right_hip_valid = True
            for j in range(0, len(people_boxes)):
                j_box = people_boxes[j]
                if j_box.conf[0] > 0.4:
                    [x1, y1, x2, y2] = j_box.xyxy[0] # floats
                    if all(x < 0 for x in left_hip_pos[i]) and point_in_box(left_hip_pos[i], (x1, y1), (x2, y2)):
                        if max(left_foot_pos[i][1], right_foot_pos[i][1]) > max(left_foot_pos[j][1], right_foot_pos[j][1]): # using feet y position to calculate who is 
                            left_hip_valid = False
                    if all(x < 0 for x in right_hip_pos[i]) and point_in_box(right_hip_pos[i], (x1, y1), (x2, y2)):
                        right_hip_valid = False
                    
                    if not left_hip_valid and not right_hip_valid:
                        break
            
            if left_hip_valid:
                pass
            elif right_hip_valid:
                pass
            else:
                pass

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow('position', frame)

videoCap.release()
cv2.destroyAllWindows()
