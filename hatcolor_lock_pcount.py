import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from utils.color_detector import threshold_color_detection


model = YOLO("yolov8m.pt")

color_bgr = {
    "Grey": (128, 128, 128),
    "White": (255, 255, 255),
    "Green": (0, 255, 0),
    "Yellow": (0, 255, 255),
    "Blue": (255, 0, 0),
    "Red": (0, 0, 255),
    "Brown": (19, 69, 139)
}

def detect_hat_color(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half_size = 12
    x1_new, y1_new = max(0, cx - half_size), max(0, cy - half_size)
    x2_new, y2_new = min(frame.shape[1], cx + half_size), min(frame.shape[0], cy + half_size)
    cropped_image = frame[y1_new:y2_new, x1_new:x2_new]
    return threshold_color_detection(cropped_image,)

cap = cv2.VideoCapture("video_path")

if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()

# Tracking dictionaries
track_hat_color = {}
pending_color_frames = {}  # new temporary color buffer

def reset_color_count():
    base_colors = ['Red', 'Green', 'Blue', 'Yellow', 'White', 'Grey', 'Brown']
    return {c: 0 for c in base_colors}

color_count = reset_color_count()

def update_color_counts():
    for key in color_count.keys():
        color_count[key] = 0
    for c in track_hat_color.values():
        if c in color_count:
            color_count[c] += 1

def draw_color_legend(frame, color_count, color_bgr):
    y_offset = 40
    for color_name, count in color_count.items():
        color = color_bgr[color_name]
        cv2.rectangle(frame, (10, y_offset - 15), (25, y_offset), color, -1)
        # text_color = (0,0,0) if color_name in ["White", "Yellow"] else (255,255,255)
        cv2.putText(frame, f"{color_name}: {count}", (35, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30


while True:
    ret, frame = cap.read()
    if not ret:
        break

    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    hat_results = model.predict(pil_frame, conf=0.03, verbose=False, classes=[1])

    hat_boxes = []
    for bbox in hat_results[0].boxes.xyxy:
        max_color, _ = detect_hat_color(frame, bbox)
        if max_color:
            hat_boxes.append((bbox.cpu().numpy().flatten(), max_color))

    person_results = model.track(frame, conf=0.4, persist=True, verbose=False, classes=[0])
    current_track_ids = set()

    for result in person_results:
        if result.boxes is not None:
            track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else [-1]*len(result.boxes)
            for box, track_id in zip(result.boxes, track_ids):
                if track_id == -1:
                    continue

                current_track_ids.add(track_id)
                px1, py1, px2, py2 = map(int, box.xyxy.cpu().numpy().flatten())

                detected_color = None
                for (hx1, hy1, hx2, hy2), color in hat_boxes:
                    if hx1 >= px1 and hy1 >= py1 and hx2 <= px2 and hy2 <= py2:
                    # if (px1 <= hx1 <= px2 and py1 <= hy2 <= py2) and (px1 <= hx2 <= px2 and py1 <= hy2 <= py2):
                        detected_color = color
                        break

                if detected_color:
                    prev_pending = pending_color_frames.get(track_id, {"color": None, "count": 0})
                    if detected_color == prev_pending["color"]:
                        prev_pending["count"] += 1
                    else:
                        prev_pending = {"color": detected_color, "count": 1}
                    pending_color_frames[track_id] = prev_pending

                    # Confirm color if seen 5 frames
                    if prev_pending["count"] >= 5:
                        current_color = track_hat_color.get(track_id)
                        if current_color != detected_color:
                            track_hat_color[track_id] = detected_color
                            print(f"✅ Confirmed {detected_color} hat for person {track_id}")
                            update_color_counts()

                # Draw bounding box
                hat_color = track_hat_color.get(track_id, "")
                box_color = color_bgr.get(hat_color, (0, 0, 0))
                cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)
                cv2.putText(frame, f"{hat_color}", (px1, py1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    # Remove disappeared IDs
    disappeared_ids = set(track_hat_color.keys()) - current_track_ids
    for tid in disappeared_ids:
        del track_hat_color[tid]
        pending_color_frames.pop(tid, None)
        update_color_counts()

    # draw_color_legend(frame, color_count, color_bgr)

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
