import requests

# НАСТРОЙКИ
IP_ADDRESS = '192.168.0.1'  # IP робота
BASE_URL = f"http://{IP_ADDRESS}"

def connect_to_robotino() -> bool:
    try:
        response = requests.get(f"{BASE_URL}/data/odometry", timeout=1)
        if response.status_code == 200:
            print("Успешное соединение с Robotino!")
            return True
        else:
            print(f"Ошибка соединения: статус {response.status_code}")
            return False
    except Exception as e:
        print(f"Ошибка соединения: {e}")
        return False

def send_velocity(vx: float, vy: float, omega: float = 0.0) -> bool:
    url = f"{BASE_URL}/data/omnidrive"

    # Преобразуем numpy типы в обычные float
    data = [float(vx), float(vy), float(omega)]

    try:
        response = requests.post(url, json=data, timeout=0.5)
        if response.status_code == 200:
            return True
        else:
            print(f"Ошибка отправки скорости: статус {response.status_code}")
            return False
    except Exception as e:
        print(f"Ошибка отправки скорости: {e}")
        return False

def stop_robot() -> bool:
    return send_velocity(0.0, 0.0, 0.0)