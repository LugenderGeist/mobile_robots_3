import os
import cv2
import math
import time
import video as vp
from robotino import (connect_to_robotino, send_velocity, stop_robot)

# РАЗМЕРЫ ПОЛЯ
FIELD_WIDTH = 220.0
FIELD_HEIGHT = 220.0

# ПРЕПЯТСТВИЯ
EDGE_MARGIN = 5
OBSTACLE_MIN_AREA = 800
OBSTACLE_MAX_AREA = 500000
THRESHOLD = 140
ROBOT_SAFETY_RADIUS = 25.0
ROBOT_RADIUS = 27.0
OBSTACLE_SAFETY_MARGIN = 8.0
PLANNING_STEP = 1.0

# УПРАВЛЕНИЕ
MAX_SPEED = 0.3
SPEED_KP = 3.2
ACC_SPEED_ERROR = 5.0
MAX_OMEGA = 0.5
ANGLE_KP = 0.5
ACC_ANGLE_ERROR = 6.0
REFERENCE_ANGLE = -90.0

EDGE_LIMIT_CM = 15.0

CORNERS_FILE = "field_corners.json"

def init_video_processor():
    vp.set_field_dimensions(FIELD_WIDTH, FIELD_HEIGHT)
    vp.set_obstacle_params(EDGE_MARGIN, OBSTACLE_MIN_AREA, OBSTACLE_MAX_AREA, THRESHOLD)
    vp.set_robot_params(ROBOT_RADIUS, ROBOT_SAFETY_RADIUS, OBSTACLE_SAFETY_MARGIN, PLANNING_STEP, EDGE_LIMIT_CM)

    if os.path.exists(CORNERS_FILE):
        vp.load_corners(CORNERS_FILE)

def mode_camera():
    init_video_processor()

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Не удалось открыть камеру!")
        return

    from planners.astar_planner import (
        create_planner, update_obstacles, draw_planning_contours,
        find_path, draw_path_on_frame, reset_path
    )

    start_point = (20.0, 20.0)
    target_point = (200.0, 200.0)

    planner = None
    current_path = None

    cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Feed", 800, 800)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Потерян кадр с камеры")
            break

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

        if current_path:
            rectified = draw_path_on_frame(planner, rectified, current_path, (0, 255, 255))

        if found:
            rectified = vp.draw_axes_2d(rectified, corners, axis_length=50)
            robot_radius_px = int(ROBOT_RADIUS / FIELD_WIDTH * rectified.shape[1])
            cv2.circle(rectified, (int(center_pixel[0]), int(center_pixel[1])), robot_radius_px, (100, 100, 100), 1)

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
        elif key == ord('r'):
            reset_path(planner)
            current_path = None
            print("Путь сброшен")

    cap.release()
    cv2.destroyAllWindows()


def mode_robot_replanning():
    if not connect_to_robotino():
        print("Не удалось подключиться к Robotino!")
        return

    init_video_processor()

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Не удалось открыть камеру!")
        return

    from planners.astar_planner import (
        create_planner, update_obstacles, draw_planning_contours,
        find_path, draw_path_on_frame, get_velocities, reset_path
    )

    start_point = (40.0, 40.0)
    target_point = (180.0, 180.0)

    planner = None
    moving = False
    rotating = False
    current_robot_pos = None
    path = None

    going_to_target = True

    last_replan_time = time.time()
    REPLAN_INTERVAL_SEC = 0.5

    replan_count = 0
    start_time = time.time()

    waiting_for_path = False

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

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Потерян кадр с камеры")
            break

        rectified = cv2.warpPerspective(frame, vp._state['H'], vp._state['output_size'])
        rectified = vp.draw_coordinate_axes(rectified, margin=20, axis_length=60)

        if vp._state['edge_limit_cm'] > 0:
            rectified = vp.draw_edge_limit(rectified)

        found, robot_id, center_pixel, _, corners = vp.detect_robot(frame)

        if found:
            robot_x, robot_y = vp.transform_coordinates(center_pixel[0], center_pixel[1])
            current_robot_pos = (robot_x, robot_y)
        else:
            current_robot_pos = None
            if moving or rotating:
                print("Робот потерян")
                stop_robot()
                moving = False
                rotating = False
                waiting_for_path = False

        if found:
            obstacles = vp.detect_obstacles(rectified, robot_center=center_pixel)
        else:
            obstacles = vp.detect_obstacles(rectified, robot_center=None)

        current_target = target_point if going_to_target else start_point
        target_name = "ЦЕЛЬ" if going_to_target else "СТАРТ"

        # Выравнивание угла
        if current_robot_pos is not None and not moving and not rotating and not waiting_for_path:
            current_angle = get_robot_angle(corners)
            delta = abs(REFERENCE_ANGLE - current_angle)
            while delta > 180:
                delta = 360 - delta

            if delta > ACC_ANGLE_ERROR:
                rotating = True

        if rotating:
            current_angle = get_robot_angle(corners)
            if rotate_to_reference_angle(current_angle):
                rotating = False

        # Периодическое обновление маршрута
        current_time = time.time()
        time_since_replan = current_time - last_replan_time

        if current_robot_pos is not None and time_since_replan >= REPLAN_INTERVAL_SEC:
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
            reset_path(planner)

            new_path = find_path(planner, current_robot_pos, current_target)

            if new_path:
                path = new_path
                last_replan_time = current_time
                replan_count += 1
                moving = True
                waiting_for_path = False
            else:
                if moving:
                    stop_robot()
                    moving = False

                if not waiting_for_path:
                    waiting_for_path = True
                path = None

        # Движение (только если есть путь и не ждем)
        if not waiting_for_path and moving and planner and path and current_robot_pos is not None:
            dist_to_target = math.hypot(current_target[0] - current_robot_pos[0],
                                        current_target[1] - current_robot_pos[1])

            if dist_to_target < ACC_SPEED_ERROR:
                stop_robot()
                going_to_target = not going_to_target
                moving = False
                rotating = False
                planner = None
                path = None
                waiting_for_path = False
                last_replan_time = time.time()
                # Продолжаем отрисовку

            else:
                vx, vy = get_velocities(
                    planner,
                    current_robot_pos[0], current_robot_pos[1],
                    max_speed=MAX_SPEED,
                    kp=SPEED_KP,
                    acc_speed_error=ACC_SPEED_ERROR
                )
                send_velocity(vx, -vy, 0.0)
        elif waiting_for_path:
            # Если ждем путь, убеждаемся что робот стоит
            send_velocity(0.0, 0.0, 0.0)

        # ========== ОТРИСОВКА ==========
        if planner is not None:
            update_obstacles(planner, obstacles)
            rectified = draw_planning_contours(planner, rectified)

        if found:
            rectified = vp.draw_axes_2d(rectified, corners, axis_length=50)
            robot_radius_px = int(ROBOT_RADIUS / FIELD_WIDTH * rectified.shape[1])
            cv2.circle(rectified, (int(center_pixel[0]), int(center_pixel[1])), robot_radius_px, (100, 100, 100), 1)

        # Рисуем путь, только если он существует
        if path and len(path) > 1:
            rectified = draw_path_on_frame(planner, rectified, path, (0, 255, 0))

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

        # Информация на экране
        info_y = 25
        cv2.putText(rectified, f"Obstacles: {len(obstacles)}", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        info_y += 25
        if current_robot_pos:
            cv2.putText(rectified,
                        f"Robot: ({current_robot_pos[0]:.1f}, {current_robot_pos[1]:.1f})",
                        (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.imshow("Robot Control", rectified)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_robot()
            break

    elapsed_time = time.time() - start_time

    cap.release()
    cv2.destroyAllWindows()

def main():
    print("1. Реальная камера")
    print("2. Управление роботом")

    choice = input("\n1 или 2? ").strip()

    if choice == '1':
        mode_camera()
    elif choice == '2':
        mode_robot_replanning()
    else:
        print("Выход")


if __name__ == "__main__":
    main()