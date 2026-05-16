import cv2
import cv2.aruco as aruco
import numpy as np
import json
import math

_state = {
    'field_width': None,
    'field_height': None,
    'corners': None,
    'H': None,
    'H_inv': None,
    'output_size': (720, 720),
    'aruco_dict': aruco.getPredefinedDictionary(aruco.DICT_6X6_250),
    'aruco_params': aruco.DetectorParameters(),
    'robot_trajectory': [],
    'edge_margin': 20,
    'obstacle_min_area': 500,
    'obstacle_max_area': 5000,
    'threshold': 100,
    'robot_safety_radius': 15.0,
    'robot_radius': 15.0,
    'obstacle_safety_margin': 5.0,
    'planning_step': 2.0,
    'edge_limit_cm': 15.0,
    'safety_mask': None,
}

# ========== ИНИЦИАЛИЗАЦИЯ ==========
def set_field_dimensions(width: float, height: float):
    _state['field_width'] = width
    _state['field_height'] = height

def set_obstacle_params(edge_margin: int = 20, min_area: int = 500,
                        max_area: int = 5000, threshold: int = 100):
    _state['edge_margin'] = edge_margin
    _state['obstacle_min_area'] = min_area
    _state['obstacle_max_area'] = max_area
    _state['threshold'] = threshold

def set_robot_params(robot_radius: float = 15.0, robot_safety_radius: float = 40.0, obstacle_safety_margin: float = 5.0,
                     planning_step: float = 2.0, edge_limit_cm: float = 15.0):
    _state['robot_radius'] = robot_radius
    _state['robot_safety_radius'] = robot_safety_radius
    _state['obstacle_safety_margin'] = obstacle_safety_margin
    _state['planning_step'] = planning_step
    _state['edge_limit_cm'] = edge_limit_cm

# ========== КАЛИБРОВКА ==========
def set_corners_manually(frame: np.ndarray) -> np.ndarray:
    corners = []
    h, w = frame.shape[:2]
    scale = min(1000 / w, 700 / h, 1.0)

    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        display_frame = cv2.resize(frame, (new_w, new_h))
    else:
        display_frame = frame.copy()
        scale = 1.0

    working_frame = display_frame.copy()

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            orig_x, orig_y = int(x / scale), int(y / scale)
            corners.append((orig_x, orig_y))
            cv2.circle(working_frame, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(working_frame, str(len(corners)), (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Select corners", working_frame)
            if len(corners) == 4:
                print("\nНажмите 'q'")

    cv2.namedWindow("Select corners", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select corners", 1000, 700)
    cv2.imshow("Select corners", working_frame)
    cv2.setMouseCallback("Select corners", mouse_callback)

    print("\nВыберите 4 угла поля: ЛВ -> ПВ -> ПН -> ЛН")
    print("После выбора нажмите 'q'\n")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') and len(corners) == 4:
            break
        elif key == 27:
            cv2.destroyAllWindows()
            raise ValueError("Выбор отменён")

    cv2.destroyAllWindows()
    return np.array(corners, dtype=np.float32)

def compute_homography(corners: np.ndarray) -> tuple:
    output_size = _state['output_size']
    dst_corners = np.array([
        [0, 0], [output_size[0], 0],
        [output_size[0], output_size[1]], [0, output_size[1]]
    ], dtype=np.float32)
    H, _ = cv2.findHomography(corners, dst_corners)
    H_inv, _ = cv2.findHomography(dst_corners, corners)
    return H, H_inv

def transform_coordinates(x_pixel: float, y_pixel: float) -> tuple:
    scale_x = _state['field_width'] / _state['output_size'][0]
    scale_y = _state['field_height'] / _state['output_size'][1]
    real_x = x_pixel * scale_x
    real_y = (_state['output_size'][1] - y_pixel) * scale_y
    return real_x, real_y

# ========== ДЕТЕКЦИЯ ==========
def detect_robot(frame: np.ndarray):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detector = cv2.aruco.ArucoDetector(_state['aruco_dict'], _state['aruco_params'])
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None and len(ids) > 0:
        marker_id = ids[0][0]
        marker_corners = corners[0][0]
        center_x = np.mean(marker_corners[:, 0])
        center_y = np.mean(marker_corners[:, 1])
        point = np.array([[[center_x, center_y]]], dtype=np.float32)
        point_rect = cv2.perspectiveTransform(point, _state['H'])
        center_x_rect = point_rect[0][0][0]
        center_y_rect = point_rect[0][0][1]
        real_x, real_y = transform_coordinates(center_x_rect, center_y_rect)
        return True, marker_id, (center_x_rect, center_y_rect), (real_x, real_y), marker_corners
    return False, -1, (0, 0), (0, 0), None


def detect_obstacles(rectified_frame: np.ndarray, robot_center: tuple = None) -> list:
    edge_margin = _state['edge_margin']
    min_area = _state['obstacle_min_area']
    max_area = _state['obstacle_max_area']
    threshold = _state['threshold']
    output_size = _state['output_size']
    field_width = _state['field_width']
    field_height = _state['field_height']
    robot_radius = _state['robot_radius']
    robot_safety_radius = _state.get('robot_safety_radius', robot_radius)
    obstacle_safety = _state['obstacle_safety_margin']
    edge_limit_cm = _state['edge_limit_cm']

    gray = cv2.cvtColor(rectified_frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    kernel = np.ones((13, 13), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    h, w = mask.shape
    mask[0:edge_margin, :] = 0
    mask[h - edge_margin:h, :] = 0
    mask[:, 0:edge_margin] = 0
    mask[:, w - edge_margin:w] = 0

    robot_center_px = None
    if robot_center is not None:
        robot_radius_px = int(robot_radius / field_width * output_size[0])
        cx = int(robot_center[0])
        cy = int(robot_center[1])
        if 0 <= cx < output_size[0] and 0 <= cy < output_size[1]:
            cv2.circle(mask, (cx, cy), robot_radius_px, 0, -1)
            robot_center_px = (cx, cy)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    all_ellipse_contours = []

    edge_limit_px = int(edge_limit_cm / field_width * output_size[0])
    min_robot_distance_px = int((robot_safety_radius + obstacle_safety) / field_width * output_size[0])

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        M = cv2.moments(contour)
        if M["m00"] != 0:
            center_x = M["m10"] / M["m00"]
            center_y = M["m01"] / M["m00"]
        else:
            continue

        if (center_x < edge_limit_px or
                center_x > output_size[0] - edge_limit_px or
                center_y < edge_limit_px or
                center_y > output_size[1] - edge_limit_px):
            continue

        if robot_center_px is not None:
            dist_to_robot = math.hypot(center_x - robot_center_px[0], center_y - robot_center_px[1])
            if dist_to_robot < min_robot_distance_px:
                continue

        # Находим минимальный ограничивающий прямоугольник (повернутый)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        # Получаем размеры прямоугольника (ширина и высота)
        width = rect[1][0]
        height = rect[1][1]
        angle = rect[2]  # угол поворота прямоугольника в градусах

        # Полуоси эллипса - половина ширины и высоты прямоугольника
        a = width / 2  # большая полуось
        b = height / 2  # малая полуось

        # Преобразуем угол в радианы
        ellipse_angle = math.radians(angle)

        # Добавляем safety margin к полуосям
        safety_px = int((robot_radius + obstacle_safety) / field_width * output_size[0])
        a_expanded = a + safety_px
        b_expanded = b + safety_px

        # Строим эллипс с расширенными полуосями
        ellipse_contour = []
        for deg in range(0, 360, 10):
            rad = np.radians(deg)
            # Параметрическое уравнение эллипса
            x = center_x + a_expanded * np.cos(rad) * np.cos(ellipse_angle) - b_expanded * np.sin(rad) * np.sin(
                ellipse_angle)
            y = center_y + a_expanded * np.cos(rad) * np.sin(ellipse_angle) + b_expanded * np.sin(rad) * np.cos(
                ellipse_angle)
            ellipse_contour.append([int(x), int(y)])

        all_ellipse_contours.append(np.array(ellipse_contour, dtype=np.int32))

    # Объединяем пересекающиеся эллипсы
    if len(all_ellipse_contours) > 0:
        merged_mask = np.zeros((output_size[1], output_size[0]), dtype=np.uint8)

        for contour in all_ellipse_contours:
            cv2.fillPoly(merged_mask, [contour], 255)

        merged_contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        obstacles = []
        safety_mask = np.zeros_like(merged_mask)
        scale_avg = (field_width / output_size[0] + field_height / output_size[1]) / 2

        for merged_contour in merged_contours:
            area = cv2.contourArea(merged_contour)
            if area < min_area:
                continue

            M = cv2.moments(merged_contour)
            if M["m00"] != 0:
                center_x_rect = M["m10"] / M["m00"]
                center_y_rect = M["m01"] / M["m00"]
            else:
                center_x_rect = 0
                center_y_rect = 0

            radius_px = int(np.sqrt(area / np.pi))
            real_x = center_x_rect * (field_width / output_size[0])
            real_y = (output_size[1] - center_y_rect) * (field_height / output_size[1])

            obstacles.append({
                'center_pixel': (center_x_rect, center_y_rect),
                'center_real': (real_x, real_y),
                'area': area,
                'radius_px': radius_px,
                'radius_cm': radius_px * scale_avg,
                'contour': merged_contour,
                'expanded_contour': merged_contour
            })

            cv2.fillPoly(safety_mask, [merged_contour], 255)

        _state['safety_mask'] = safety_mask
        return obstacles

    return []

# ========== ОТРИСОВКА ==========
def draw_axes_2d(frame: np.ndarray, marker_corners: np.ndarray, axis_length: float = 60) -> np.ndarray:
    if len(marker_corners.shape) == 3:
        marker_corners = marker_corners[0]
    pts = marker_corners.astype(np.float32).reshape(-1, 1, 2)
    rectified_corners = cv2.perspectiveTransform(pts, _state['H'])
    rectified_corners = rectified_corners.reshape(-1, 2)
    center = np.mean(rectified_corners, axis=0)
    cx, cy = int(center[0]), int(center[1])

    dx = rectified_corners[3][0] - rectified_corners[0][0]
    dy = rectified_corners[3][1] - rectified_corners[0][1]
    length = np.sqrt(dx * dx + dy * dy)
    if length > 0:
        dx = dx / length * axis_length
        dy = dy / length * axis_length

    ux = rectified_corners[0][0] - rectified_corners[1][0]
    uy = rectified_corners[0][1] - rectified_corners[1][1]
    length = np.sqrt(ux * ux + uy * uy)
    if length > 0:
        ux = ux / length * axis_length
        uy = uy / length * axis_length

    x_end = (cx + int(dx), cy + int(dy))
    cv2.arrowedLine(frame, (cx, cy), x_end, (0, 0, 255), 2, tipLength=0.3)
    cv2.putText(frame, "X", (x_end[0] + 5, x_end[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    y_end = (cx - int(ux), cy - int(uy))
    cv2.arrowedLine(frame, (cx, cy), y_end, (0, 255, 0), 2, tipLength=0.3)
    cv2.putText(frame, "Y", (y_end[0] + 5, y_end[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.circle(frame, (cx, cy), 4, (0, 0, 0), -1)
    return frame

def draw_coordinate_axes(frame: np.ndarray, margin: int = 20, axis_length: int = 50) -> np.ndarray:
    h, w = frame.shape[:2]
    origin_x = margin
    origin_y = h - margin
    x_end = (origin_x + axis_length, origin_y)
    cv2.arrowedLine(frame, (origin_x, origin_y), x_end, (0, 0, 255), 2, tipLength=0.2)
    cv2.putText(frame, "X", (x_end[0] + 5, x_end[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    y_end = (origin_x, origin_y - axis_length)
    cv2.arrowedLine(frame, (origin_x, origin_y), y_end, (0, 255, 0), 2, tipLength=0.2)
    cv2.putText(frame, "Y", (y_end[0] + 5, y_end[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.circle(frame, (origin_x, origin_y), 4, (0, 0, 0), -1)
    return frame

def draw_edge_limit(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    edge_limit_px = int(_state['edge_limit_cm'] / _state['field_width'] * w)
    margin = max(edge_limit_px, 1)
    dash_length = 15
    gap_length = 15
    for x in range(margin, w - margin, dash_length + gap_length):
        cv2.line(frame, (x, margin), (min(x + dash_length, w - margin), margin), (0, 0, 0), 2)
        y_bottom = h - margin
        cv2.line(frame, (x, y_bottom), (min(x + dash_length, w - margin), y_bottom), (0, 0, 0), 2)
    for y in range(margin, h - margin, dash_length + gap_length):
        cv2.line(frame, (margin, y), (margin, min(y + dash_length, h - margin)), (0, 0, 0), 2)
        x_right = w - margin
        cv2.line(frame, (x_right, y), (x_right, min(y + dash_length, h - margin)), (0, 0, 0), 2)
    return frame

def save_corners(corners_file: str = "field_corners.json"):
    if _state['corners'] is None:
        return
    data = {
        'corners': _state['corners'].tolist(),
        'field_width': _state['field_width'],
        'field_height': _state['field_height']
    }
    with open(corners_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Углы сохранены в {corners_file}")

def load_corners(corners_file: str) -> bool:
    try:
        with open(corners_file, 'r') as f:
            data = json.load(f)
            _state['corners'] = np.array(data['corners'], dtype=np.float32)
            _state['field_width'] = data.get('field_width', _state['field_width'])
            _state['field_height'] = data.get('field_height', _state['field_height'])
            _state['H'], _state['H_inv'] = compute_homography(_state['corners'])
            print(f"Углы загружены из {corners_file}")
            return True
    except Exception as e:
        print(f"Не удалось загрузить углы: {e}")
        return False

# ========== ОСНОВНЫЕ ФУНКЦИИ ==========
def reset_trajectory():
    _state['robot_trajectory'] = []


def process_camera_feed(camera_id: int = 0, single_frame: bool = False):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Не удалось открыть камеру {camera_id}")
        return

    if _state['corners'] is None:
        print("\nНе заданы углы поля!")
        ret, first_frame = cap.read()
        if ret:
            _state['corners'] = set_corners_manually(first_frame)
            _state['H'], _state['H_inv'] = compute_homography(_state['corners'])
            save_corners("field_corners.json")
        cap.release()
        return

    cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Feed", 800, 800)

    user_point = None
    user_point_real = None
    current_robot_pos = None
    planner = None
    current_path = None  # Сохраняем текущий путь

    def mouse_callback(event, x, y, flags, param):
        nonlocal user_point, user_point_real, planner, current_path
        if event == cv2.EVENT_LBUTTONDOWN:
            user_point = (x, y)
            real_x, real_y = transform_coordinates(x, y)
            user_point_real = (real_x, real_y)
            current_path = None  # Сбрасываем путь при новой цели
            if planner:
                from planners.astar_planner import reset_path
                reset_path(planner)

    cv2.setMouseCallback("Camera Feed", mouse_callback)

    frame_count = 0
    paused = False
    _state['robot_trajectory'] = []

    from planners.astar_planner import (
        create_planner, update_obstacles, draw_planning_contours,
        find_path, draw_path_on_frame
    )

    import math

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            rectified = cv2.warpPerspective(frame, _state['H'], _state['output_size'])
            rectified = draw_coordinate_axes(rectified, margin=20, axis_length=60)

            if _state['edge_limit_cm'] > 0:
                rectified = draw_edge_limit(rectified)

            found, robot_id, center_pixel, center_real, marker_corners = detect_robot(frame)

            robot_real_x = 0.0
            robot_real_y = 0.0
            if found:
                robot_real_x, robot_real_y = transform_coordinates(center_pixel[0], center_pixel[1])
                current_robot_pos = (robot_real_x, robot_real_y)
                _state['robot_trajectory'].append({
                    'frame': frame_count,
                    'x_pixel': center_pixel[0],
                    'y_pixel': center_pixel[1],
                    'x_real': robot_real_x,
                    'y_real': robot_real_y
                })

            if found:
                obstacles = detect_obstacles(rectified, robot_center=center_pixel)
            else:
                obstacles = detect_obstacles(rectified, robot_center=None)

            if planner is None:
                planner = create_planner(
                    field_width=_state['field_width'],
                    field_height=_state['field_height'],
                    step=_state['planning_step'],
                    robot_radius=_state['robot_radius'],
                    obstacle_safety=_state['obstacle_safety_margin'],
                    edge_limit_cm=_state['edge_limit_cm']
                )

            update_obstacles(planner, obstacles)
            rectified = draw_planning_contours(planner, rectified)

            # Строим путь только если есть цель и робот
            if user_point_real is not None and found and current_robot_pos is not None:
                # Проверяем, достиг ли робот цели
                dist_to_goal = math.hypot(current_robot_pos[0] - user_point_real[0],
                                          current_robot_pos[1] - user_point_real[1])

                # Если робот достиг цели
                if dist_to_goal < 5.0:
                    if user_point_real is not None:
                        user_point_real = None
                        user_point = None
                        current_path = None
                        if planner:
                            from planners.astar_planner import reset_path
                            reset_path(planner)
                else:
                    # Если нет пути, строим новый
                    if current_path is None or len(current_path) == 0:
                        current_path = find_path(planner, current_robot_pos, user_point_real)

            if current_path is not None and len(current_path) > 1:
                rectified = draw_path_on_frame(planner, rectified, current_path, (0, 255, 0))
            if found:
                cx = int(center_pixel[0])
                cy = int(center_pixel[1])
                robot_radius_px = int(_state['robot_radius'] / _state['field_width'] * _state['output_size'][0])
                cv2.circle(rectified, (cx, cy), robot_radius_px, (100, 100, 100), 1)
                rectified = draw_axes_2d(rectified, marker_corners, axis_length=50)

            if user_point is not None:
                cv2.circle(rectified, user_point, 8, (0, 255, 0), -1)

            # Информация на экране
            info_y = 25
            cv2.putText(rectified, f"Obstacles: {len(obstacles)}", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            info_y += 25

            if found:
                cv2.putText(rectified, f"Robot: ({robot_real_x:.1f}, {robot_real_y:.1f})",
                            (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                info_y += 25

            if user_point_real is not None:
                cv2.putText(rectified, f"Target: ({user_point_real[0]:.1f}, {user_point_real[1]:.1f})",
                            (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                info_y += 25

            if rectified.shape[1] > 800 or rectified.shape[0] > 800:
                scale = min(800 / rectified.shape[1], 800 / rectified.shape[0])
                new_w = int(rectified.shape[1] * scale)
                new_h = int(rectified.shape[0] * scale)
                display = cv2.resize(rectified, (new_w, new_h))
            else:
                display = rectified

            cv2.imshow("Camera Feed", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if single_frame and frame_count >= 1:
            break

    cap.release()
    cv2.destroyAllWindows()