import cv2
import numpy as np
from typing import List, Tuple
import math
import heapq

def create_planner(field_width: float, field_height: float, step: float,
                   robot_radius: float, obstacle_safety: float,
                   edge_limit_cm: float) -> dict:
    grid_width = int(field_width / step) + 1
    grid_height = int(field_height / step) + 1

    planner = {
        'field_width': field_width,
        'field_height': field_height,
        'step': step,
        'robot_radius': robot_radius,
        'obstacle_safety': obstacle_safety,
        'edge_limit_cm': edge_limit_cm,
        'grid_width': grid_width,
        'grid_height': grid_height,
        'obstacles': [],
        'obstacle_map': np.zeros((grid_height, grid_width), dtype=np.uint8),
        'path': []
    }
    return planner

def update_obstacles(planner: dict, obstacles: List[dict]):
    planner['obstacles'] = obstacles
    planner['obstacle_map'].fill(0)

    for obs in obstacles:
        center_x, center_y = obs['center_real']
        total_radius = obs['radius_cm'] + planner['robot_radius'] + planner['obstacle_safety']

        step = planner['step']
        grid_width = planner['grid_width']
        grid_height = planner['grid_height']

        cell_x = int(center_x / step)
        cell_y = int(center_y / step)
        cell_radius = int(total_radius / step) + 1

        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                grid_x = cell_x + dx
                grid_y = cell_y + dy
                if (0 <= grid_x < grid_width and 0 <= grid_y < grid_height):
                    if dx * dx + dy * dy <= cell_radius * cell_radius:
                        planner['obstacle_map'][grid_y, grid_x] = 1

def is_cell_safe(planner: dict, grid_x: int, grid_y: int) -> bool:
    if not (0 <= grid_x < planner['grid_width'] and 0 <= grid_y < planner['grid_height']):
        return False

    cx = (grid_x + 0.5) * planner['step']
    cy = (grid_y + 0.5) * planner['step']

    if (cx < planner['edge_limit_cm'] or
            cx > planner['field_width'] - planner['edge_limit_cm'] or
            cy < planner['edge_limit_cm'] or
            cy > planner['field_height'] - planner['edge_limit_cm']):
        return False

    for obs in planner['obstacles']:
        obs_x, obs_y = obs['center_real']
        obs_radius = obs['radius_cm']
        dist = math.hypot(cx - obs_x, cy - obs_y)
        if dist < obs_radius + planner['robot_radius']:
            return False

    return True

def heuristic(x: int, y: int, goal_x: int, goal_y: int, step: float) -> float:
    dx = (x - goal_x) * step
    dy = (y - goal_y) * step
    return math.hypot(dx, dy)

def get_step_cost(dx: int, dy: int, step: float) -> float:
    if dx != 0 and dy != 0:
        return math.sqrt(2) * step
    return step

def world_to_grid(planner: dict, x: float, y: float) -> Tuple[int, int]:
    step = planner['step']
    grid_x = int(x / step)
    grid_y = int(y / step)
    grid_x = max(0, min(grid_x, planner['grid_width'] - 1))
    grid_y = max(0, min(grid_y, planner['grid_height'] - 1))
    return grid_x, grid_y

def grid_to_world(planner: dict, grid_x: int, grid_y: int) -> Tuple[float, float]:
    step = planner['step']
    return (grid_x + 0.5) * step, (grid_y + 0.5) * step

def find_path(planner: dict, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
    start_grid = world_to_grid(planner, start[0], start[1])
    goal_grid = world_to_grid(planner, goal[0], goal[1])

    if not is_cell_safe(planner, start_grid[0], start_grid[1]):
        print("    Стартовая позиция небезопасна")
        return []

    if not is_cell_safe(planner, goal_grid[0], goal_grid[1]):
        print("    Целевая позиция небезопасна")
        return []

    moves = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1), (0, 1),
             (1, -1), (1, 0), (1, 1)]

    INF = float('inf')
    g_score = {}
    parent = {}

    start_node = (start_grid[0], start_grid[1])
    g_score[start_node] = 0

    h = heuristic(start_grid[0], start_grid[1], goal_grid[0], goal_grid[1], planner['step'])
    pq = [(h, start_grid[0], start_grid[1])]

    while pq:
        _, x, y = heapq.heappop(pq)
        current = (x, y)

        if current == (goal_grid[0], goal_grid[1]):
            path = []
            curr = current
            while curr in parent:
                path.append(grid_to_world(planner, curr[0], curr[1]))
                curr = parent[curr]
            path.append(grid_to_world(planner, start_grid[0], start_grid[1]))
            path.reverse()
            planner['path'] = path
            return path

        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)

            if not (0 <= nx < planner['grid_width'] and 0 <= ny < planner['grid_height']):
                continue
            if not is_cell_safe(planner, nx, ny):
                continue

            step_cost = get_step_cost(dx, dy, planner['step'])
            tentative_g = g_score[current] + step_cost

            if tentative_g < g_score.get(neighbor, INF):
                parent[neighbor] = current
                g_score[neighbor] = tentative_g
                h = heuristic(nx, ny, goal_grid[0], goal_grid[1], planner['step'])
                heapq.heappush(pq, (tentative_g + h, nx, ny))

    print("    Путь не найден")
    return []

def get_velocities(planner: dict, current_x: float, current_y: float,
                   max_speed: float, kp: float, acc_speed_error: float) -> Tuple[float, float]:
    path = planner['path']
    if not path or len(path) < 2:
        return 0.0, 0.0

    min_dist = float('inf')
    nearest_idx = 0
    for i, point in enumerate(path):
        px, py = point
        dist = math.hypot(px - current_x, py - current_y)
        if dist < min_dist:
            min_dist = dist
            nearest_idx = i

    target_idx = min(nearest_idx + 3, len(path) - 1)
    target_x, target_y = path[target_idx]

    error_x = target_x - current_x
    error_y = target_y - current_y
    error_distance = math.hypot(error_x, error_y)

    min_speed_ms = 0.03

    max_speed_cm = max_speed * 100.0
    speed_cm = min(kp * error_distance, max_speed_cm)

    final_goal = path[-1]
    dist_to_final = math.hypot(final_goal[0] - current_x, final_goal[1] - current_y)
    if dist_to_final > acc_speed_error:
        speed_cm = max(speed_cm, min_speed_ms * 100.0)

    if error_distance > 0:
        vx = (error_x / error_distance) * (speed_cm / 100.0)
        vy = (error_y / error_distance) * (speed_cm / 100.0)
    else:
        vx, vy = 0.0, 0.0

    return vx, -vy

def draw_planning_contours(planner: dict, frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    step = planner['step']
    field_width = planner['field_width']
    field_height = planner['field_height']

    obstacle_map_uint8 = (planner['obstacle_map'] * 255).astype(np.uint8)
    contours, _ = cv2.findContours(obstacle_map_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if len(contour) < 4:
            continue
        pixel_contour = []
        for point in contour:
            gx, gy = point[0]
            x = int(gx * step / field_width * w)
            y = int(h - (gy * step / field_height * h))
            pixel_contour.append([x, y])
        if len(pixel_contour) > 2:
            pixel_contour = np.array(pixel_contour, dtype=np.int32)
            cv2.polylines(frame, [pixel_contour], True, (0, 255, 255), 2)
    return frame

def draw_path_on_frame(planner: dict, frame: np.ndarray, path: List[Tuple[float, float]],
                       color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    if not path or len(path) < 2:
        return frame

    h, w = frame.shape[:2]
    field_width = planner['field_width']
    field_height = planner['field_height']

    points = []
    for real_x, real_y in path:
        x_px = int(real_x / field_width * w)
        y_px = int(h - (real_y / field_height * h))
        points.append((x_px, y_px))

    for i in range(len(points) - 1):
        cv2.line(frame, points[i], points[i + 1], color, 3)

    return frame