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

    # Преобразуем в список точек (унифицируем формат)
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

    # Находим центр масс
    center_x = sum(p[0] for p in points) / len(points)
    center_y = sum(p[1] for p in points) / len(points)

    # Сортируем точки по углу относительно центра
    def get_angle(point):
        return math.atan2(point[1] - center_y, point[0] - center_x)

    sorted_points = sorted(points, key=get_angle)

    # Группируем точки по квадрантам и берем самую удаленную в каждом
    quadrants = [[] for _ in range(4)]
    for point in sorted_points:
        angle = get_angle(point)
        if -math.pi / 4 <= angle < math.pi / 4:
            quadrants[0].append(point)  # право
        elif math.pi / 4 <= angle < 3 * math.pi / 4:
            quadrants[1].append(point)  # верх
        elif -3 * math.pi / 4 <= angle < -math.pi / 4:
            quadrants[2].append(point)  # низ
        else:
            quadrants[3].append(point)  # лево

    # В каждом квадранте выбираем точку, максимально удаленную от центра
    farthest_corners = []
    for quadrant in quadrants:
        if quadrant:
            farthest = max(quadrant, key=lambda p: math.hypot(p[0] - center_x, p[1] - center_y))
            farthest_corners.append(farthest)

    # Если получилось меньше 4 углов, добавляем оставшиеся самые удаленные
    if len(farthest_corners) < 4:
        remaining = [p for p in points if p not in farthest_corners]
        remaining.sort(key=lambda p: math.hypot(p[0] - center_x, p[1] - center_y), reverse=True)
        farthest_corners.extend(remaining[:4 - len(farthest_corners)])

    # Преобразуем в формат для отрисовки
    result = []
    for point in farthest_corners:
        result.append([int(point[0]), int(point[1])])

    return np.array(result, dtype=np.int32)


def find_polygon_from_contour(contour, output_size, corner_quality=0.01, min_distance=20):
    """Находит углы контура и строит полигон (прямоугольник из 4 углов)"""
    # Находим центр контура
    M = cv2.moments(contour)
    if M["m00"] != 0:
        center_x = M["m10"] / M["m00"]
        center_y = M["m01"] / M["m00"]
    else:
        return None, None

    # Создаем маску для текущего контура
    contour_mask = np.zeros((output_size[1], output_size[0]), dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, -1)

    # Находим углы с помощью детектора Shi-Tomasi
    corners = cv2.goodFeaturesToTrack(contour_mask, maxCorners=20, qualityLevel=corner_quality,
                                      minDistance=min_distance)

    if corners is None or len(corners) < 4:
        # Если углов мало, используем аппроксимацию контура
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) >= 4:
            # Берем 4 наиболее удаленных угла из аппроксимации
            approx_points = []
            for point in approx:
                approx_points.append((float(point[0][0]), float(point[0][1])))

            # Находим центр
            ax = sum(p[0] for p in approx_points) / len(approx_points)
            ay = sum(p[1] for p in approx_points) / len(approx_points)

            # Сортируем по углу и берем 4 угла
            approx_points.sort(key=lambda p: math.atan2(p[1] - ay, p[0] - ax))

            # Берем каждую 4-ю точку для прямоугольника
            step = max(1, len(approx_points) // 4)
            polygon_points = [approx_points[i] for i in range(0, len(approx_points), step)][:4]

            polygon_contour = []
            for point in polygon_points:
                polygon_contour.append([int(point[0]), int(point[1])])
            return np.array(polygon_contour, dtype=np.int32), (center_x, center_y)
        else:
            return None, None
    else:
        # Находим 4 самых удаленных друг от друга угла
        farthest_corners = find_farthest_corners(corners)

        if len(farthest_corners) < 4:
            return None, None

        # Сортируем углы по часовой стрелке
        center = np.mean(farthest_corners, axis=0)

        def sort_by_angle(point):
            angle = math.atan2(point[1] - center[1], point[0] - center[0])
            return angle

        sorted_corners = sorted(farthest_corners, key=sort_by_angle)

        return np.array(sorted_corners, dtype=np.int32), (center_x, center_y)


def expand_polygon(polygon, center_x, center_y, safety_px):
    """Расширяет полигон от центра на safety_px пикселей"""
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
        print("\nНет калибровки для камеры!")
        cap.release()
        return

    cv2.namedWindow("Obstacle Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Obstacle Detection", 800, 800)

    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mask", 400, 400)

    cv2.namedWindow("Parameters", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Parameters", 400, 400)

    # Основные параметры детекции
    cv2.createTrackbar("Threshold", "Parameters", 165, 255, nothing)
    cv2.createTrackbar("Min Area", "Parameters", 800, 5000, nothing)
    cv2.createTrackbar("Edge Margin", "Parameters", 5, 100, nothing)

    # Параметры морфологии
    cv2.createTrackbar("Open Kernel", "Parameters", 5, 20, nothing)
    cv2.createTrackbar("Close Kernel", "Parameters", 5, 20, nothing)

    # Параметры безопасности
    cv2.createTrackbar("Safety Margin", "Parameters", 0, 50, nothing)
    cv2.createTrackbar("Robot Mask", "Parameters", 1, 1, nothing)

    # Параметры детекции углов
    cv2.createTrackbar("Corner Quality", "Parameters", 10, 100, nothing)  # qualityLevel * 0.01
    cv2.createTrackbar("Min Distance", "Parameters", 20, 100, nothing)

    try:
        with open(PARAMS_FILE, "r") as f:
            saved = json.load(f)
            cv2.setTrackbarPos("Threshold", "Parameters", saved.get('threshold', 165))
            cv2.setTrackbarPos("Min Area", "Parameters", saved.get('min_area', 800))
            cv2.setTrackbarPos("Edge Margin", "Parameters", saved.get('edge_margin', 5))
            cv2.setTrackbarPos("Open Kernel", "Parameters", saved.get('open_kernel', 5))
            cv2.setTrackbarPos("Close Kernel", "Parameters", saved.get('close_kernel', 5))
            cv2.setTrackbarPos("Safety Margin", "Parameters", saved.get('safety_margin', 0))
            cv2.setTrackbarPos("Robot Mask", "Parameters", saved.get('robot_mask', 1))
            cv2.setTrackbarPos("Corner Quality", "Parameters", saved.get('corner_quality', 10))
            cv2.setTrackbarPos("Min Distance", "Parameters", saved.get('min_distance', 20))
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
        open_kernel_size = cv2.getTrackbarPos("Open Kernel", "Parameters")
        close_kernel_size = cv2.getTrackbarPos("Close Kernel", "Parameters")
        safety_margin = cv2.getTrackbarPos("Safety Margin", "Parameters")
        use_robot_mask = cv2.getTrackbarPos("Robot Mask", "Parameters")
        corner_quality = cv2.getTrackbarPos("Corner Quality", "Parameters") / 100.0
        min_distance = cv2.getTrackbarPos("Min Distance", "Parameters")

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
        mask[0:edge_margin, :] = 0
        mask[h - edge_margin:h, :] = 0
        mask[:, 0:edge_margin] = 0
        mask[:, w - edge_margin:w] = 0

        # Маска робота
        robot_center = None
        if use_robot_mask:
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            aruco_params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
            corners, ids, _ = detector.detectMarkers(frame)

            if ids is not None and len(ids) > 0:
                marker_corners = corners[0][0]
                center_x = np.mean(marker_corners[:, 0])
                center_y = np.mean(marker_corners[:, 1])
                point = np.array([[[center_x, center_y]]], dtype=np.float32)
                point_rect = cv2.perspectiveTransform(point, H)
                robot_center = (int(point_rect[0][0][0]), int(point_rect[0][0][1]))

                robot_radius_px = int(ROBOT_RADIUS / FIELD_WIDTH * 720)
                cv2.circle(mask, robot_center, robot_radius_px, 0, -1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result = rectified.copy()
        obstacle_count = 0
        safety_px = int((ROBOT_RADIUS + safety_margin) / FIELD_WIDTH * 720)

        # Обновляем параметры детектора углов
        corner_params = dict(maxCorners=8, qualityLevel=corner_quality, minDistance=min_distance)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # Находим полигон из углов
                polygon, center = find_polygon_from_contour(contour, (w, h), corner_quality, min_distance)

                if polygon is not None and len(polygon) >= 3:
                    # Рисуем зеленым оригинальный полигон (линии между углами)
                    cv2.polylines(result, [polygon], True, (0, 255, 0), 2)

                    # Рисуем углы красными кружками
                    for point in polygon:
                        cv2.circle(result, (point[0], point[1]), 5, (0, 0, 255), -1)

                    # Создаем и рисуем расширенный полигон (желтый)
                    if safety_px > 0:
                        expanded = expand_polygon(polygon, center[0], center[1], safety_px)
                        cv2.polylines(result, [expanded], True, (0, 255, 255), 2)
                        overlay = result.copy()
                        cv2.fillPoly(overlay, [expanded], (0, 255, 255))
                        cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)

                    # Рисуем центр препятствия
                    cv2.circle(result, (int(center[0]), int(center[1])), 6, (255, 0, 0), -1)
                    obstacle_count += 1
                else:
                    # Если не удалось найти углы, рисуем обычный контур
                    cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = M["m10"] / M["m00"]
                        cy = M["m01"] / M["m00"]
                        cv2.circle(result, (int(cx), int(cy)), 6, (255, 0, 0), -1)
                    obstacle_count += 1

        # Рисуем положение робота
        if robot_center is not None:
            robot_radius_px = int(ROBOT_RADIUS / FIELD_WIDTH * 720)
            cv2.circle(result, robot_center, robot_radius_px, (100, 100, 100), 2)
            cv2.circle(result, robot_center, 5, (255, 255, 0), -1)

        # Рисуем границы поля
        cv2.rectangle(result, (edge_margin, edge_margin),
                      (w - edge_margin, h - edge_margin), (255, 255, 255), 2)

        if paused:
            cv2.putText(result, "PAUSED", (result.shape[1] - 100, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Информация
        cv2.putText(mask_colored, f"Obstacles: {obstacle_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if obstacle_count > 0 else (0, 0, 255), 1)
        cv2.putText(mask_colored, f"Open: {open_kernel}x{open_kernel}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(mask_colored, f"Close: {close_kernel}x{close_kernel}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        info_y = 30
        cv2.putText(result, f"Frame: {current_frame}", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(result, f"Threshold: {threshold_val}, Min Area: {min_area}",
                    (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result, f"Open: {open_kernel}x{open_kernel}, Close: {close_kernel}x{close_kernel}",
                    (10, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result, f"Safety Margin: {safety_margin} cm, Safety PX: {safety_px}",
                    (10, info_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result, f"Robot Mask: {'ON' if use_robot_mask else 'OFF'}",
                    (10, info_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result, f"Corner Quality: {corner_quality:.2f}, Min Dist: {min_distance}",
                    (10, info_y + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result, f"Obstacles: {obstacle_count}",
                    (10, info_y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if obstacle_count > 0 else (0, 0, 255), 1)

        # Легенда
        legend_y = result.shape[0] - 95
        cv2.putText(result, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result, "Green: Polygon (corners connected)", (10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 255, 0), 1)
        cv2.putText(result, "Red: Detected corners", (10, legend_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(result, "Yellow: Expanded (robot + safety)", (10, legend_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 255, 255), 1)
        cv2.putText(result, "Blue: Center", (10, legend_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        cv2.putText(result, "Cyan: Robot", (10, legend_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        cv2.imshow("Obstacle Detection", result)
        cv2.imshow("Mask", mask_colored)

        key = cv2.waitKey(1 if not paused else 0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            params = {
                'threshold': threshold_val,
                'min_area': min_area,
                'edge_margin': edge_margin,
                'open_kernel': open_kernel_size,
                'close_kernel': close_kernel_size,
                'safety_margin': safety_margin,
                'robot_mask': use_robot_mask,
                'corner_quality': cv2.getTrackbarPos("Corner Quality", "Parameters"),
                'min_distance': min_distance
            }
            with open(PARAMS_FILE, "w") as f:
                json.dump(params, f, indent=2)
            print(f"Сохранено")
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