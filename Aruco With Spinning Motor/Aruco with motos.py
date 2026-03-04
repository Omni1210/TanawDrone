import cv2
import numpy as np
import math
import time
import msvcrt
import threading
from pymavlink import mavutil

# ─── MAVLINK CONNECTION ────────────────────────
master = mavutil.mavlink_connection('COM13', baud=57600)
print("Waiting for heartbeat...")
master.wait_heartbeat()
print(f"Connected! System: {master.target_system}")

master.mav.param_set_send(
    master.target_system,
    master.target_component,
    b'ARMING_CHECK',
    0,
    mavutil.mavlink.MAV_PARAM_TYPE_INT32
)
time.sleep(2)

master.mav.set_mode_send(
    master.target_system,
    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
    0
)
time.sleep(2)

# ─── SHARED STATE ──────────────────────────────
manual_throttle = 1200
MIN_THROTTLE    = 1100
MAX_THROTTLE    = 1800
STEP            = 50
spinning        = False
landing         = False
auto_mode       = False

marker_x        = 0.0
marker_y        = 0.0
marker_distance = 0.0
marker_detected = False
lock = threading.Lock()

# ─── POSITION PID (Roll/Pitch) ─────────────────
POS_KP = 100.0
POS_KI = 0.0
POS_KD = 20.0

class PID:
    def __init__(self, kp, ki, kd, min_out, max_out):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_out    = min_out
        self.max_out    = max_out
        self.prev_error = 0.0
        self.integral   = 0.0
        self.last_time  = time.time()

    def update(self, error):
        now = time.time()
        dt  = now - self.last_time
        if dt <= 0:
            dt = 0.01
        self.integral    += error * dt
        self.integral     = max(-100, min(100, self.integral))
        derivative        = (error - self.prev_error) / dt
        output            = (self.kp * error +
                             self.ki * self.integral +
                             self.kd * derivative)
        self.prev_error   = error
        self.last_time    = now
        return max(self.min_out, min(self.max_out, output))

    def reset(self):
        self.prev_error = 0.0
        self.integral   = 0.0
        self.last_time  = time.time()

roll_pid  = PID(POS_KP, POS_KI, POS_KD, -200, 200)
pitch_pid = PID(POS_KP, POS_KI, POS_KD, -200, 200)

# ─── THROTTLE FROM DISTANCE ────────────────────
def distance_to_throttle(distance):
    # 0.1m → 1100, 1.0m+ → 1500
    throttle = 1100 + (distance - 0.1) * 444
    return int(max(MIN_THROTTLE, min(MAX_THROTTLE, throttle)))

# ─── MAVLINK FUNCTIONS ─────────────────────────
def send_rc(roll, pitch, throttle_val, yaw=1500):
    master.mav.rc_channels_override_send(
        master.target_system,
        master.target_component,
        roll, pitch, throttle_val, yaw,
        0, 0, 0, 0
    )

def arm():
    print("▶ Arming...")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0
    )
    time.sleep(3)
    print("▶ Armed! Motors ready.")

def disarm():
    for _ in range(5):
        send_rc(1500, 1500, 1000)
        time.sleep(0.05)
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 0, 0, 0, 0, 0, 0, 0
    )
    print("⏹ Disarmed!")

def land():
    global manual_throttle, spinning, landing
    print("🛬 Landing sequence started...")
    current = manual_throttle
    while current > 1000:
        current = max(current - 50, 1000)
        send_rc(1500, 1500, current)
        print(f"  🛬 Throttle: {current}")
        time.sleep(0.5)
    spinning = False
    landing  = False
    disarm()
    print("✅ Landed safely!")

# ─── ARUCO THREAD ──────────────────────────────
def aruco_thread():
    global marker_x, marker_y, marker_distance, marker_detected

    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    camera_matrix = np.array([
        [445, 0,   320],
        [0,   445, 240],
        [0,   0,   1  ]
    ], dtype=np.float32)

    dist_coeffs = np.zeros((4, 1))
    MARKER_SIZE = 0.10

    obj_points = np.array([
        [-MARKER_SIZE/2,  MARKER_SIZE/2, 0],
        [ MARKER_SIZE/2,  MARKER_SIZE/2, 0],
        [ MARKER_SIZE/2, -MARKER_SIZE/2, 0],
        [-MARKER_SIZE/2, -MARKER_SIZE/2, 0]
    ], dtype=np.float32)

    frame_count = 0
    last_frame  = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 2 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                for i, corner in enumerate(corners):
                    success, rvec, tvec = cv2.solvePnP(
                        obj_points, corner[0], camera_matrix, dist_coeffs
                    )

                    if success:
                        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

                        x = tvec[0][0]
                        y = tvec[1][0]
                        z = tvec[2][0]

                        auto_thr = distance_to_throttle(z)

                        with lock:
                            marker_x        = x
                            marker_y        = y
                            marker_distance = z
                            marker_detected = True

                        marker_id = ids[i][0]
                        top_left  = (int(corner[0][0][0]), int(corner[0][0][1]))

                        if x > 0.05:
                            direction = "Marker LEFT"
                            dir_color = (0, 165, 255)
                        elif x < -0.05:
                            direction = "Marker RIGHT"
                            dir_color = (0, 165, 255)
                        else:
                            direction = "Marker CENTERED"
                            dir_color = (0, 255, 0)

                        # ─── ON SCREEN ─────────
                        cv2.putText(frame, f"ID: {marker_id}",
                            (top_left[0], top_left[1] - 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame, f"Distance: {z:.3f} m",
                            (top_left[0], top_left[1] - 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                        cv2.putText(frame, f"Auto Throttle: {auto_thr}",
                            (top_left[0], top_left[1] - 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 100), 2)
                        cv2.putText(frame, f"X:{x:.2f}  Y:{y:.2f}",
                            (top_left[0], top_left[1] - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        cv2.putText(frame, direction,
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, dir_color, 2)

                        mode_text  = "🤖 AUTO MODE" if auto_mode else "🕹 MANUAL (Press A)"
                        mode_color = (0, 255, 0) if auto_mode else (0, 165, 255)
                        cv2.putText(frame, mode_text,
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)

            else:
                with lock:
                    marker_detected = False
                cv2.putText(frame, "No marker detected",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "🕹 MANUAL MODE",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            last_frame = frame.copy()

        if last_frame is not None:
            cv2.imshow("ArUco Hover", last_frame)
        else:
            cv2.imshow("ArUco Hover", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ─── START ARUCO THREAD ────────────────────────
t = threading.Thread(target=aruco_thread, daemon=True)
t.start()

arm()
spinning = True

print("\n✅ Ready! Controls:")
print("  A   - Toggle AUTO mode")
print("  W   - Throttle Up (manual)")
print("  S   - Throttle Down (manual)")
print("  Q   - Stop & Disarm")
print("  E   - Arm & Start Again")
print("  L   - Land (Gentle)")
print("  T   - End Program\n")

while True:
    with lock:
        detected = marker_detected
        mx       = marker_x
        my       = marker_y
        mz       = marker_distance

    if spinning and not landing:
        if auto_mode and detected:
            # ─── DISTANCE → THROTTLE ───────────
            auto_thr = distance_to_throttle(mz)

            # ─── POSITION PID (Roll/Pitch) ─────
            roll_adjust  = roll_pid.update(mx)
            pitch_adjust = pitch_pid.update(my)
            roll_out     = int(max(1300, min(1700, 1500 + roll_adjust)))
            pitch_out    = int(max(1300, min(1700, 1500 + pitch_adjust)))

            send_rc(roll_out, pitch_out, auto_thr)

            print(f"  🤖 Dist:{mz:.2f}m → Thr:{auto_thr} "
                  f"Roll:{roll_out} Pitch:{pitch_out}")

        elif auto_mode and not detected:
            print("  ⚠ No marker! Holding...")
            send_rc(1500, 1500, manual_throttle)
        else:
            send_rc(1500, 1500, manual_throttle)

    elif not landing:
        send_rc(1500, 1500, 1000)

    if msvcrt.kbhit():
        key = msvcrt.getch().decode('utf-8').lower()

        if key == 'a':
            auto_mode = not auto_mode
            if auto_mode:
                roll_pid.reset()
                pitch_pid.reset()
                print("  🤖 AUTO MODE ON — throttle follows distance!")
            else:
                print("  🕹 MANUAL MODE")

        elif key == 'w':
            if not auto_mode and spinning and not landing:
                manual_throttle = min(manual_throttle + STEP, MAX_THROTTLE)
                print(f"  ⬆ Throttle: {manual_throttle}")
            elif auto_mode:
                print("  ⚠ In AUTO mode! Press A to switch to manual.")
            else:
                print("  ⚠ Motors not running!")

        elif key == 's':
            if not auto_mode and spinning and not landing:
                manual_throttle = max(manual_throttle - STEP, MIN_THROTTLE)
                print(f"  ⬇ Throttle: {manual_throttle}")
            elif auto_mode:
                print("  ⚠ In AUTO mode! Press A to switch to manual.")
            else:
                print("  ⚠ Motors not running!")

        elif key == 'q':
            if landing:
                print("  ⚠ Landing in progress!")
            elif spinning:
                auto_mode       = False
                spinning        = False
                manual_throttle = 1200
                disarm()
            else:
                print("  ⚠ Motors already stopped!")

        elif key == 'e':
            if landing:
                print("  ⚠ Landing in progress!")
            elif not spinning:
                master.mav.set_mode_send(
                    master.target_system,
                    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                    0
                )
                time.sleep(1)
                arm()
                spinning  = True
                landing   = False
                auto_mode = False
            else:
                print("  ⚠ Motors already running!")

        elif key == 'l':
            if spinning and not landing:
                auto_mode = False
                landing   = True
                spinning  = False
                land()
            elif landing:
                print("  ⚠ Already landing!")
            else:
                print("  ⚠ Motors not running!")

        elif key == 't':
            print("\nShutting down...")
            auto_mode = False
            if spinning or landing:
                land()
            else:
                disarm()
            print("✅ Goodbye!")
            break

    time.sleep(0.1)