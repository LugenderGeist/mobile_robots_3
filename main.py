import os
import cv2
import math
import time
import video as vp
from robotino import (connect_to_robotino, send_velocity, stop_robot)
from comparison import save_results

# РАЗМЕРЫ ПОЛЯ
FIELD_WIDTH = 220.0
FIELD_HEIGHT = 220.0

# ПРЕПЯТСТВИЯ
EDGE_MARGIN = 5
OBSTACLE_MIN_AREA = 800
OBSTACLE_MAX_AREA = 500000
THRESHOLD = 145
ROBOT_RADIUS = 34.0
OBSTACLE_SAFETY_MARGIN = 0.0
PLANNING_STEP = 2.0

# УПРАВЛЕНИЕ
MAX_SPEED = 0.2
SPEED_KP = 2.2
ACC_SPEED_ERROR = 5.0
MAX_OMEGA = 0.5
ANGLE_KP = 0.5
ACC_ANGLE_ERROR = 7.0
REFERENCE_ANGLE = -90.0

EDGE_LIMIT_CM = 15.0

CORNERS_FILE = "field_corners.json"

def init_video_processor():
    vp.set_field_dimensions(FIELD_WIDTH, FIELD_HEIGHT)
    vp.set_obstacle_params(EDGE_MARGIN, OBSTACLE_MIN_AREA, OBSTACLE_MAX_AREA, THRESHOLD)
    vp.set_robot_params(ROBOT_RADIUS, OBSTACLE_SAFETY_MARGIN, PLANNING_STEP, EDGE_LIMIT_CM)

    if os.path.exists(CORNERS_FILE):
        vp.load_corners(CORNERS_FILE)

def mode_camera():
    init_video_processor()

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Не удалось открыть камеру!")
        return

    from planners.greedy_planner import (
        create_planner, update_obstacles, draw_planning_contours,
        find_path, draw_path_on_frame
    )

    # ЗАДАЁМ КООРДИНАТЫ СТАРТОВОЙ И ЦЕЛЕВОЙ ТОЧЕК
    start_point = (20.0, 20.0)  # стартовая точка (см)
    target_point = (200.0, 200.0)  # целевая точка (см)

    planner = None
    frame_count = 0

    cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Feed", 800, 800)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Потерян кадр с камеры")
            break
        frame_count += 1

        rectified = cv2.warpPerspective(frame, vp._state['H'], vp._state['output_size'])
        rectified = vp.draw_coordinate_axes(rectified, margin=20, axis_length=60)

        if vp._state['edge_limit_cm'] > 0:
            rectified = vp.draw_edge_limit(rectified)

        found, robot_id, center_pixel, _, corners = vp.detect_robot(frame)

        current_robot_pos = None
        if found:
            robot_x, robot_y = vp.transform_coordinates(center_pixel[0], center_pixel[1])
            current_robot_pos = (robot_x, robot_y)

        if found:
            obstacles = vp.detect_obstacles(rectified, robot_center=center_pixel)
        else:
            obstacles = vp.detect_obstacles(rectified, robot_center=None)

        if planner is None:
            planner = create_planner(
                field_width=FIELD_WIDTH,
                field_height=FIELD_HEIGHT,
                step=PLANNING_STEP,
                robot_radius=ROBOT_RADIUS,
                obstacle_safety=OBSTACLE_SAFETY_MARGIN,
                edge_limit_cm=EDGE_LIMIT_CM
            )

        update_obstacles(planner, obstacles)
        rectified = draw_planning_contours(planner, rectified)

        if found:
            rectified = vp.draw_axes_2d(rectified, corners, axis_length=50)
            robot_radius_px = int(ROBOT_RADIUS / FIELD_WIDTH * rectified.shape[1])
            cv2.circle(rectified, (int(center_pixel[0]), int(center_pixel[1])), robot_radius_px, (100, 100, 100), 1)

        # Строим и отображаем путь от стартовой точки к целевой
        path = find_path(planner, start_point, target_point)
        if path:
            rectified = draw_path_on_frame(planner, rectified, path, (0, 255, 255))

        # Отрисовка стартовой точки
        tx = int(start_point[0] / FIELD_WIDTH * rectified.shape[1])
        ty = int(rectified.shape[0] - (start_point[1] / FIELD_HEIGHT * rectified.shape[0]))
        cv2.circle(rectified, (tx, ty), 8, (0, 255, 0), -1)
        cv2.circle(rectified, (tx, ty), 12, (0, 255, 0), 2)
        cv2.putText(rectified, "START", (tx + 10, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Отрисовка целевой точки
        tx = int(target_point[0] / FIELD_WIDTH * rectified.shape[1])
        ty = int(rectified.shape[0] - (target_point[1] / FIELD_HEIGHT * rectified.shape[0]))
        cv2.circle(rectified, (tx, ty), 8, (255, 0, 0), -1)
        cv2.circle(rectified, (tx, ty), 12, (255, 0, 0), 2)
        cv2.putText(rectified, "GOAL", (tx + 10, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        info_y = 25
        cv2.putText(rectified, f"Obstacles: {len(obstacles)}", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        info_y += 25
        if current_robot_pos:
            cv2.putText(rectified, f"Robot: ({current_robot_pos[0]:.1f}, {current_robot_pos[1]:.1f})",
                        (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

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

    cap.release()
    cv2.destroyAllWindows()

def mode_robot():
    path_start = None  # путь к стартовой точке
    path_target = None  # путь к целевой точке

    if not connect_to_robotino():
        print("Не удалось подключиться к Robotino!")
        return

    init_video_processor()

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Не удалось открыть камеру!")
        return

    from planners.greedy_planner import (
        create_planner, update_obstacles, draw_planning_contours,
        find_path, draw_path_on_frame, get_velocities
    )

    # ЗАДАЁМ КООРДИНАТЫ СТАРТОВОЙ И ЦЕЛЕВОЙ ТОЧЕК
    start_point = (20.0, 20.0)  # стартовая точка (см)
    target_point = (180.0, 180.0)  # целевая точка (см)

    planner = None
    moving = False
    rotating = False
    at_start = False
    current_robot_pos = None
    path = None

    # ПЕРЕМЕННЫЕ ДЛЯ СБОРА ДАННЫХ
    planned_path = None
    actual_trajectory = []
    search_time_ms = 0
    travel_start_time = None
    speed_log = []

    def get_robot_angle(marker_corners) -> float:
        if marker_corners is None:
            return 0.0
        corner_points = marker_corners.reshape(4, 2)
        dx = corner_points[1][0] - corner_points[0][0]
        dy = corner_points[1][1] - corner_points[0][1]
        return math.degrees(math.atan2(dy, dx))

    def rotate_to_reference_angle(current_angle: float) -> bool:
        delta = REFERENCE_ANGLE - current_angle
        while delta > 180:
            delta -= 360
        while delta < -180:
            delta += 360

        print(f"current={current_angle:.1f}°, target={REFERENCE_ANGLE:.1f}°, delta={delta:.1f}°")

        if abs(delta) < ACC_ANGLE_ERROR:
            return True

        omega = -math.radians(delta) * ANGLE_KP
        max_omega = MAX_OMEGA
        omega = max(-max_omega, min(omega, max_omega))

        if abs(omega) < 0.05 and abs(delta) > ACC_ANGLE_ERROR:
            omega = 0.2 if delta > 0 else -0.2

        send_velocity(0.0, 0.0, omega)
        return False

    cv2.namedWindow("Robot Control", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Robot Control", 800, 800)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Потерян кадр с камеры")
            break
        frame_count += 1

        rectified = cv2.warpPerspective(frame, vp._state['H'], vp._state['output_size'])
        rectified = vp.draw_coordinate_axes(rectified, margin=20, axis_length=60)

        if vp._state['edge_limit_cm'] > 0:
            rectified = vp.draw_edge_limit(rectified)

        found, robot_id, center_pixel, _, corners = vp.detect_robot(frame)

        if found:
            robot_x, robot_y = vp.transform_coordinates(center_pixel[0], center_pixel[1])
            current_robot_pos = (robot_x, robot_y)

            # ЗАПИСЫВАЕМ ТРАЕКТОРИЮ
            if travel_start_time is not None:
                actual_trajectory.append((robot_x, robot_y, time.time()))
        else:
            current_robot_pos = None
            if moving or rotating:
                print("Робот потерян")
                stop_robot()
                moving = False
                rotating = False

        if found:
            obstacles = vp.detect_obstacles(rectified, robot_center=center_pixel)
        else:
            obstacles = vp.detect_obstacles(rectified, robot_center=None)

        # ========== ДВИЖЕНИЕ К СТАРТОВОЙ ТОЧКЕ ==========
        if not at_start:
            if current_robot_pos is not None:
                dist_to_start = math.hypot(start_point[0] - current_robot_pos[0], start_point[1] - current_robot_pos[1])

                if dist_to_start < ACC_SPEED_ERROR:
                    stop_robot()
                    at_start = True
                    moving = False
                    rotating = False
                    planner = None
                    path = None
                    continue

                # Выравнивание угла
                current_angle = get_robot_angle(corners)
                delta = abs(REFERENCE_ANGLE - current_angle)
                while delta > 180:
                    delta = 360 - delta

                if delta > ACC_ANGLE_ERROR and not rotating:
                    rotating = True
                    continue

                if rotating:
                    if rotate_to_reference_angle(current_angle):
                        rotating = False
                    continue

                # Создаём планировщик и строим путь к стартовой точке
                if planner is None:
                    planner = create_planner(
                        field_width=FIELD_WIDTH,
                        field_height=FIELD_HEIGHT,
                        step=PLANNING_STEP,
                        robot_radius=ROBOT_RADIUS,
                        obstacle_safety=OBSTACLE_SAFETY_MARGIN,
                        edge_limit_cm=EDGE_LIMIT_CM
                    )
                    update_obstacles(planner, obstacles)
                    path = find_path(planner, current_robot_pos, start_point)
                    if path:
                        moving = True
                    else:
                        print("Не удалось построить путь к стартовой точке")

                # Движение
                if moving and planner and path:
                    vx, vy = get_velocities(
                        planner,
                        current_robot_pos[0], current_robot_pos[1],
                        max_speed=MAX_SPEED,
                        kp=SPEED_KP,
                        acc_speed_error=ACC_SPEED_ERROR
                    )
                    send_velocity(vx, -vy, 0.0)

                    if frame_count % 30 == 0:
                        print(f"vx={vx:.3f}, vy={vy:.3f}, до старта={dist_to_start:.1f} см")

            if path_start is None:
                path_start = find_path(planner, current_robot_pos, start_point)
                if path_start:
                    moving = True

            if path_start:
                rectified = draw_path_on_frame(planner, rectified, path_start, (0, 255, 255))

        # ========== ДВИЖЕНИЕ К ЦЕЛЕВОЙ ТОЧКЕ ==========
        if at_start and current_robot_pos is not None:
            dist_to_target = math.hypot(target_point[0] - current_robot_pos[0], target_point[1] - current_robot_pos[1])

            if dist_to_target < ACC_SPEED_ERROR:
                travel_time = time.time() - travel_start_time if travel_start_time else 0
                stop_robot()

                # СОХРАНЯЕМ РЕЗУЛЬТАТЫ
                save_results(
                    "comparison_report.txt",
                    planned_path,
                    actual_trajectory,
                    search_time_ms,
                    travel_time,
                    max(speed_log) if speed_log else 0,
                    sum(speed_log) / len(speed_log) if speed_log else 0
                )
                break

            # Выравнивание угла
            current_angle = get_robot_angle(corners)
            delta = abs(REFERENCE_ANGLE - current_angle)
            while delta > 180:
                delta = 360 - delta

            if delta > ACC_ANGLE_ERROR and not rotating:
                rotating = True
                continue

            if rotating:
                if rotate_to_reference_angle(current_angle):
                    rotating = False
                continue

            # Создаём планировщик и строим путь к целевой точке
            if planner is None:
                planner = create_planner(
                    field_width=FIELD_WIDTH,
                    field_height=FIELD_HEIGHT,
                    step=PLANNING_STEP,
                    robot_radius=ROBOT_RADIUS,
                    obstacle_safety=OBSTACLE_SAFETY_MARGIN,
                    edge_limit_cm=EDGE_LIMIT_CM
                )
                update_obstacles(planner, obstacles)
                search_start = time.time()
                path = find_path(planner, current_robot_pos, target_point)
                search_time_ms = (time.time() - search_start) * 1000
                if path:
                    planned_path = path
                    if travel_start_time is None:
                        travel_start_time = time.time()
                    moving = True
                else:
                    print("Не удалось построить путь к целевой точке")

            if moving and planner and path:
                vx, vy = get_velocities(
                    planner,
                    current_robot_pos[0], current_robot_pos[1],
                    max_speed=MAX_SPEED,
                    kp=SPEED_KP,
                    acc_speed_error=ACC_SPEED_ERROR
                )
                send_velocity(vx, -vy, 0.0)

                speed = math.hypot(vx, vy) * 100.0
                speed_log.append(speed)

                # На этапе 2 (после достижения стартовой точки):
                if path_target is None:
                    path_target = find_path(planner, current_robot_pos, target_point)
                    if path_target:
                        moving = True

                # Отрисовка на этапе 2:
                if path_target:
                    rectified = draw_path_on_frame(planner, rectified, path_target, (0, 255, 255))

                if frame_count % 30 == 0:
                    print(f"vx={vx:.3f}, vy={vy:.3f}, до цели={dist_to_target:.1f} см")

        if planner is not None:
            update_obstacles(planner, obstacles)
            rectified = draw_planning_contours(planner, rectified)

        # Отрисовка робота
        if found:
            rectified = vp.draw_axes_2d(rectified, corners, axis_length=50)
            robot_radius_px = int(ROBOT_RADIUS / FIELD_WIDTH * rectified.shape[1])
            cv2.circle(rectified, (int(center_pixel[0]), int(center_pixel[1])), robot_radius_px, (100, 100, 100), 1)

        # Отрисовка стартовой и целевой точек
        tx = int(start_point[0] / FIELD_WIDTH * rectified.shape[1])
        ty = int(rectified.shape[0] - (start_point[1] / FIELD_HEIGHT * rectified.shape[0]))
        cv2.circle(rectified, (tx, ty), 8, (0, 255, 0), -1)
        cv2.circle(rectified, (tx, ty), 12, (0, 255, 0), 2)
        cv2.putText(rectified, "START", (tx + 10, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        tx = int(target_point[0] / FIELD_WIDTH * rectified.shape[1])
        ty = int(rectified.shape[0] - (target_point[1] / FIELD_HEIGHT * rectified.shape[0]))
        cv2.circle(rectified, (tx, ty), 8, (255, 0, 0), -1)
        cv2.circle(rectified, (tx, ty), 12, (255, 0, 0), 2)
        cv2.putText(rectified, "GOAL", (tx + 10, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Информация
        info_y = 25
        cv2.putText(rectified, f"Obstacles: {len(obstacles)}", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        info_y += 25
        if current_robot_pos:
            cv2.putText(rectified, f"Robot: ({current_robot_pos[0]:.1f}, {current_robot_pos[1]:.1f})",
                        (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imshow("Robot Control", rectified)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_robot()
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    print("1. Реальная камера")
    print("2. Управление роботом")

    choice = input("\n1 или 2? ").strip()

    if choice == '1':
        mode_camera()
    elif choice == '2':
        mode_robot()
    else:
        print("Выход")

if __name__ == "__main__":
    main()