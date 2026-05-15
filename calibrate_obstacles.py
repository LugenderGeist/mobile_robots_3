import cv2
import numpy as np
import json
import math

# ========== НАСТРОЙКИ ==========
CAMERA_ID = 1
CORNERS_FILE = "field_corners.json"
PARAMS_FILE = "obstacle_params_camera.json"
FIELD_WIDTH = 220.0
FIELD_HEIGHT = 220.0
ROBOT_RADIUS = 34.0
OBSTACLE_SAFETY_MARGIN = 0.0
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


def find_farthest_corners(corners):
    if len(corners) <= 4:
        return corners

    points = []
    for corner in corners:
        if isinstance(corner, np.ndarray):
            if len(corner) == 2:
                points.append((float(corner[0]), float(corner[1])))
            elif len(corner) == 1 and len(corner[0]) == 2:
                points.append((float(corner[0][0]), float(corner[0][1])))
            else:
                points.append((float(corner[0]), float(corner[1])))
        elif len(corner) == 2:
            points.append((float(corner[0]), float(corner[1])))
        else:
            points.append((float(corner[0][0]), float(corner[0][1])))

    center_x = sum(p[0] for p in points) / len(points)
    center_y = sum(p[1] for p in points) / len(points)

    def get_angle(point):
        return math.atan2(point[1] - center_y, point[0] - center_x)

    sorted_points = sorted(points, key=get_angle)

    quadrants = [[] for _ in range(4)]
    for point in sorted_points:
        angle = get_angle(point)
        if -math.pi / 4 <= angle < math.pi / 4:
            quadrants[0].append(point)
        elif math.pi / 4 <= angle < 3 * math.pi / 4:
            quadrants[1].append(point)
        elif -3 * math.pi / 4 <= angle < -math.pi / 4:
            quadrants[2].append(point)
        else:
            quadrants[3].append(point)

    farthest_corners = []
    for quadrant in quadrants:
        if quadrant:
            farthest = max(quadrant, key=lambda p: math.hypot(p[0] - center_x, p[1] - center_y))
            farthest_corners.append(farthest)

    if len(farthest_corners) < 4:
        remaining = [p for p in points if p not in farthest_corners]
        remaining.sort(key=lambda p: math.hypot(p[0] - center_x, p[1] - center_y), reverse=True)
        farthest_corners.extend(remaining[:4 - len(farthest_corners)])

    result = []
    for point in farthest_corners:
        result.append([int(point[0]), int(point[1])])

    return np.array(result, dtype=np.int32)


def find_polygon_from_contour(contour, output_size, corner_quality=0.01, min_distance=20):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        center_x = M["m10"] / M["m00"]
        center_y = M["m01"] / M["m00"]
    else:
        return None, None

    contour_mask = np.zeros((output_size[1], output_size[0]), dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, -1)

    corners = cv2.goodFeaturesToTrack(contour_mask, maxCorners=20, qualityLevel=corner_quality,
                                      minDistance=min_distance)

    if corners is None or len(corners) < 4:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) >= 4:
            approx_points = []
            for point in approx:
                approx_points.append((float(point[0][0]), float(point[0][1])))

            ax = sum(p[0] for p in approx_points) / len(approx_points)
            ay = sum(p[1] for p in approx_points) / len(approx_points)

            approx_points.sort(key=lambda p: math.atan2(p[1] - ay, p[0] - ax))

            step = max(1, len(approx_points) // 4)
            polygon_points = [approx_points[i] for i in range(0, len(approx_points), step)][:4]

            polygon_contour = []
            for point in polygon_points:
                polygon_contour.append([int(point[0]), int(point[1])])
            return np.array(polygon_contour, dtype=np.int32), (center_x, center_y)
        else:
            return None, None
    else:
        farthest_corners = find_farthest_corners(corners)

        if len(farthest_corners) < 4:
            return None, None

        center = np.mean(farthest_corners, axis=0)

        def sort_by_angle(point):
            angle = math.atan2(point[1] - center[1], point[0] - center[0])
            return angle

        sorted_corners = sorted(farthest_corners, key=sort_by_angle)

        return np.array(sorted_corners, dtype=np.int32), (center_x, center_y)


def expand_polygon(polygon, center_x, center_y, safety_px):
    expanded = []
    for point in polygon:
        px, py = point
        dx = px - center_x
        dy = py - center_y
        dist = np.sqrt(dx * dx + dy * dy)
        if dist > 0:
            scale = (dist + safety_px) / dist
            new_x = center_x + dx * scale
            new_y = center_y + dy * scale
            expanded.append([int(new_x), int(new_y)])
        else:
            expanded.append([int(px), int(py)])
    return np.array(expanded, dtype=np.int32)


def main():
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"Не удалось открыть камеру {CAMERA_ID}")
        return

    H, corners = load_homography()
    if H is None:
        print("Нет калибровки для камеры!")
        cap.release()
        return

    cv2.namedWindow("Obstacle Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Obstacle Detection", 800, 800)

    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mask", 400, 400)

    cv2.namedWindow("Parameters", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Parameters", 400, 150)

    # Только нужные трекбары
    cv2.createTrackbar("Threshold", "Parameters", 165, 255, nothing)
    cv2.createTrackbar("Open Kernel", "Parameters", 5, 20, nothing)
    cv2.createTrackbar("Close Kernel", "Parameters", 5, 20, nothing)

    try:
        with open(PARAMS_FILE, "r") as f:
            saved = json.load(f)
            cv2.setTrackbarPos("Threshold", "Parameters", saved.get('threshold', 165))
            cv2.setTrackbarPos("Open Kernel", "Parameters", saved.get('open_kernel', 5))
            cv2.setTrackbarPos("Close Kernel", "Parameters", saved.get('close_kernel', 5))
            print("Загружены сохранённые параметры")
    except:
        pass

    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Потерян кадр с камеры")
                continue

        threshold_val = cv2.getTrackbarPos("Threshold", "Parameters")
        open_kernel_size = cv2.getTrackbarPos("Open Kernel", "Parameters")
        close_kernel_size = cv2.getTrackbarPos("Close Kernel", "Parameters")

        rectified = cv2.warpPerspective(frame, H, (720, 720))
        gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)

        # Морфологические операции
        open_kernel = max(1, open_kernel_size if open_kernel_size % 2 == 1 else open_kernel_size + 1)
        close_kernel = max(1, close_kernel_size if close_kernel_size % 2 == 1 else close_kernel_size + 1)

        kernel_open = np.ones((open_kernel, open_kernel), np.uint8)
        kernel_close = np.ones((close_kernel, close_kernel), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        h, w = mask.shape

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result = rectified.copy()
        obstacle_count = 0
        safety_px = int((ROBOT_RADIUS + OBSTACLE_SAFETY_MARGIN) / FIELD_WIDTH * 720)
        corner_quality = 0.01
        min_distance = 20

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 800:  # min_area по умолчанию
                polygon, center = find_polygon_from_contour(contour, (w, h), corner_quality, min_distance)

                if polygon is not None and len(polygon) >= 3:
                    cv2.polylines(result, [polygon], True, (0, 255, 0), 2)

                    for point in polygon:
                        cv2.circle(result, (point[0], point[1]), 5, (0, 0, 255), -1)

                    if safety_px > 0:
                        expanded = expand_polygon(polygon, center[0], center[1], safety_px)
                        cv2.polylines(result, [expanded], True, (0, 255, 255), 2)
                        overlay = result.copy()
                        cv2.fillPoly(overlay, [expanded], (0, 255, 255))
                        cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)

                    cv2.circle(result, (int(center[0]), int(center[1])), 6, (255, 0, 0), -1)
                    obstacle_count += 1

        if paused:
            cv2.putText(result, "PAUSED", (result.shape[1] - 100, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(mask_colored, f"Obstacles: {obstacle_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if obstacle_count > 0 else (0, 0, 255), 1)
        cv2.putText(mask_colored, f"Open: {open_kernel}x{open_kernel}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(mask_colored, f"Close: {close_kernel}x{close_kernel}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        info_y = 30
        cv2.putText(result, f"Threshold: {threshold_val}", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result, f"Open: {open_kernel}x{open_kernel}, Close: {close_kernel}x{close_kernel}",
                    (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result, f"Obstacles: {obstacle_count}",
                    (10, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if obstacle_count > 0 else (0, 0, 255), 1)

        cv2.imshow("Obstacle Detection", result)
        cv2.imshow("Mask", mask_colored)

        key = cv2.waitKey(1 if not paused else 0) & 0xFF

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()