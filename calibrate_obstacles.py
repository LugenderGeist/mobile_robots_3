import cv2
import numpy as np
import json

# ========== НАСТРОЙКИ ==========
CAMERA_ID = 1
CORNERS_FILE = "field_corners.json"
PARAMS_FILE = "obstacle_params_camera.json"
# =================================

def nothing(x):
    pass

def load_homography():
    try:
        with open(CORNERS_FILE, 'r') as f:
            data = json.load(f)
            corners = np.array(data['corners'], dtype=np.float32)
            dst = np.array([[0, 0], [720, 0], [720, 720], [0, 720]], dtype=np.float32)
            H, _ = cv2.findHomography(corners, dst)
            print(f"Загружены углы поля из {CORNERS_FILE}")
            return H, corners
    except Exception as e:
        print(f"Не удалось загрузить углы: {e}")
        return None, None

def main():
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"Не удалось открыть камеру {CAMERA_ID}")
        return

    H, corners = load_homography()
    if H is None:
        print("\nНет калибровки для камеры!")
        cap.release()
        return

    cv2.namedWindow("Obstacle Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Obstacle Detection", 800, 800)

    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mask", 400, 400)

    cv2.namedWindow("Parameters", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Parameters", 400, 200)

    cv2.createTrackbar("Threshold", "Parameters", 220, 255, nothing)
    cv2.createTrackbar("Min Area", "Parameters", 500, 5000, nothing)
    cv2.createTrackbar("Edge Margin", "Parameters", 20, 100, nothing)

    try:
        with open(PARAMS_FILE, "r") as f:
            saved = json.load(f)
            cv2.setTrackbarPos("Threshold", "Parameters", saved.get('threshold', 220))
            cv2.setTrackbarPos("Min Area", "Parameters", saved.get('min_area', 500))
            cv2.setTrackbarPos("Edge Margin", "Parameters", saved.get('edge_margin', 20))
            print("Загружены сохранённые параметры")
    except:
        pass

    current_frame = 0
    total_frames = 1000
    paused = False
    cv2.createTrackbar("Frame", "Parameters", 0, total_frames - 1, nothing)

    while True:
        if not paused:
            current_frame = cv2.getTrackbarPos("Frame", "Parameters")
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        threshold_val = cv2.getTrackbarPos("Threshold", "Parameters")
        min_area = cv2.getTrackbarPos("Min Area", "Parameters")
        edge_margin = cv2.getTrackbarPos("Edge Margin", "Parameters")

        rectified = cv2.warpPerspective(frame, H, (720, 720))
        gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        h, w = mask.shape
        mask[0:edge_margin, :] = 0
        mask[h - edge_margin:h, :] = 0
        mask[:, 0:edge_margin] = 0
        mask[:, w - edge_margin:w] = 0

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result = rectified.copy()
        obstacle_count = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                cv2.drawContours(result, [contour], -1, (0, 0, 255), 2)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(result, (cx, cy), 6, (0, 255, 0), -1)
                obstacle_count += 1

        cv2.rectangle(result, (edge_margin, edge_margin),
                      (w - edge_margin, h - edge_margin), (0, 255, 255), 2)

        if paused:
            cv2.putText(result, "PAUSED", (result.shape[1] - 100, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(mask_colored, f"Obstacles: {obstacle_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if obstacle_count > 0 else (0, 0, 255), 1)

        info_y = 30
        cv2.putText(result, f"Frame: {current_frame}", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(result, f"Threshold: {threshold_val}, Min Area: {min_area}",
                    (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result, f"Obstacles: {obstacle_count}",
                    (10, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if obstacle_count > 0 else (0, 0, 255), 1)

        cv2.imshow("Obstacle Detection", result)
        cv2.imshow("Mask", mask_colored)

        key = cv2.waitKey(1 if not paused else 0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            params = {
                'threshold': threshold_val,
                'min_area': min_area,
                'edge_margin': edge_margin
            }
            with open(PARAMS_FILE, "w") as f:
                json.dump(params, f, indent=2)
            print(f"Сохранено: Threshold={threshold_val}, Min Area={min_area}, Edge Margin={edge_margin}")
        elif key == ord('p'):
            paused = not paused
            print("Пауза" if paused else "Продолжение")
        elif key == 81 or key == 2424832:
            current_frame = max(0, current_frame - 30)
            cv2.setTrackbarPos("Frame", "Parameters", current_frame)
            paused = False
        elif key == 83 or key == 2555904:
            current_frame = min(total_frames - 1, current_frame + 30)
            cv2.setTrackbarPos("Frame", "Parameters", current_frame)
            paused = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()