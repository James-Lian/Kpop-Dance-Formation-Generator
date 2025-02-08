import cv2
from ultralytics import YOLO # object tracking
import mediapipe as mp # pose estimation

from depth import *
# from perspective import find_vanishing_point
from PIL import Image
import numpy as np

vanishing_point = (562, 391)

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
    v_path = input("gimme a path: ")
    videoCap = cv2.VideoCapture(v_path)

# videoCap = cv2.VideoCapture(0)
yolo = YOLO('yolov8s.pt')

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

midasObj = Midas(ModelType.MIDAS_SMALL)
midasObj.useCUDA()
midasObj.transform()

relative_distance = {}
horizontal_distance = {} # regardless of perspective
left_foot_pos = {}
right_foot_pos = {}
left_hip_pos = {}
right_hip_pos = {}

frame_num = 0

while videoCap.isOpened():
    ret, frame = videoCap.read()
    if not ret:
        continue

    results = yolo.track(frame, stream=True, persist=True)
    # Object detection and visualization code

    #depth detection
    depth = midasObj.predict(frame)

    people_boxes = []
    for result in results:
        classes_names = result.names
        for box in result.boxes:
            if box.conf[0] > 0.4:
                if classes_names[int(box.cls[0])] == "person":
                    people_boxes.append(box)

    for i in range(0, len(people_boxes)):
        box = people_boxes[i]
        if box.id == None:
            continue
        id = int(box.id.item())
        if id not in relative_distance:
            relative_distance[id] = {frame_num: {}}
        else:
            relative_distance[id][frame_num] = {}
        
        if id not in horizontal_distance:
            horizontal_distance[id] = {}


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
                left_foot_pos[id] = (int(left_foot.x * width), int(left_foot.y * height))
            except:
                left_foot_pos[id] = (-1, -1)

            try:
                right_foot = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
                right_foot_pos[id] = (int(right_foot.x * width), int(right_foot.y * height))
            except:
                right_foot_pos[id] = (-1, -1)

            try:
                left_hip = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                left_hip_pos[id] = (int(left_hip.x * width), int(left_hip.y * height))
            except:
                left_hip_pos[id] = (-1, -1)

            try:
                right_hip = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
                right_hip_pos[id] = (int(right_hip.x * width), int(right_hip.y * height))
            except:
                right_hip_pos[id] = (-1, -1)

    
    for i in range(0, len(people_boxes)): # runs after all poses have been calculated
        box = people_boxes[i]
        if box.id == None:
            continue
        id = int(box.id.item())

        if box.conf[0] > 0.4:

            # distance calculation based on foot position

            # relative distance calculation based on depth estimation
            ## determine that key pose landmarks are not obstructed by other people:
            left_hip_valid = True
            right_hip_valid = True

            for j in range(0, len(people_boxes)):
                j_box = people_boxes[j]
                if j_box.conf[0] > 0.4:
                    [x1, y1, x2, y2] = j_box.xyxy[0] # floats
                    if all(x < 0 for x in left_hip_pos[id]) and point_in_box(left_hip_pos[id], (x1, y1), (x2, y2)):
                        if max(left_foot_pos[id][1], right_foot_pos[id][1]) > max(left_foot_pos[j][1], right_foot_pos[j][1]): # using feet y position to calculate who is 
                            left_hip_valid = False
                    if all(x < 0 for x in right_hip_pos[id]) and point_in_box(right_hip_pos[id], (x1, y1), (x2, y2)):
                        right_hip_valid = False
                    
                    if not left_hip_valid and not right_hip_valid:
                        break
            
            if left_hip_valid and right_hip_valid:
                relative_distance[id][frame_num]['depth-estimation'] = depth[int((left_hip_pos[id][1]+right_hip_pos[id][1])/2)][int((left_hip_pos[id][0]+right_hip_pos[id][0])/2)]
                
            if left_hip_valid:
                relative_distance[id][frame_num]['depth-estimation'] = depth[left_hip_pos[id][1]][left_hip_pos[id][0]]
            elif right_hip_valid:
                relative_distance[id][frame_num]['depth-estimation'] = depth[right_hip_pos[id][1]][right_hip_pos[id][0]]
            
            # if mediapipe detected both left and right foot
            if all(x > 0 for x in left_foot_pos[id]) and all(x > 0 for x in right_foot_pos[id]):
                relative_distance[id][frame_num]['foot-pos'] = max(left_foot_pos[id][1], right_foot_pos[id][1])

                # people further away from the camera appear closer to the center even though they may be at the same x-coordinate
                # drawing a line from vanishing point to the feet, where this line intersects with the bottom of the frame is the person's world x-coordinates
                # (y2-y1) / (x2-x1)
                m = (vanishing_point[1] - max(left_foot_pos[id][1], right_foot_pos[id][1])) / (vanishing_point[0] - (left_foot_pos[id][0] + right_foot_pos[id][0])/2) 
                b = vanishing_point[1] - (m * vanishing_point[0])
                # sub frame.shape[1] as y in y=mx+b
                x = (frame.shape[1] - b)/m
                horizontal_distance[id][frame_num] = x

            elif all(x > 0 for x in left_foot_pos[id]):
                relative_distance[id][frame_num]['foot-pos'] = left_foot_pos[id][1]

                m = (vanishing_point[1] - left_foot_pos[id][1]) / (vanishing_point[0] - left_foot_pos[id][0])
                b = vanishing_point[1] - (m * vanishing_point[0])
                x = (frame.shape[1] - b)/m
                horizontal_distance[id][frame_num] = x

            elif all(x > 0 for x in right_foot_pos[id]):
                relative_distance[id][frame_num]['foot-pos'] = right_foot_pos[id][1]

                m = (vanishing_point[1] - right_foot_pos[id][1]) / (vanishing_point[0] - right_foot_pos[id][0])
                b = vanishing_point[1] - (m * vanishing_point[0])
                x = (frame.shape[1] - b)/m
                horizontal_distance[id][frame_num] = x

    map = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    for object in relative_distance:
        try:
            cv2.circle(map, (int(horizontal_distance[object][frame_num]/3), int(relative_distance[object][frame_num]['depth-estimation']/255*frame.shape[1])), 80, (255, 255, 255), 2)
        except:
            pass
    cv2.imshow('position', map)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_num += 1
    print(len(relative_distance))

videoCap.release()
cv2.destroyAllWindows()