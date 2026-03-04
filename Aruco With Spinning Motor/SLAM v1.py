import cv2
import numpy as np
import math
import time
import msvcrt
import threading
import json
import os
from pymavlink import mavutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

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
mapping_mode    = False
nav_mode        = False

marker_x        = 0.0
marker_y        = 0.0
marker_distance = 0.0
marker_detected = False
current_marker_id = -1
lock = threading.Lock()

# ─── MAP ───────────────────────────────────────
MAP_FILE     = "marker_map.json"
marker_map   = {}       # { marker_id: {x, y, z, visits} }
drone_pos    = [0.0, 0.0, 0.0]  # Estimated drone position
nav_target   = None     # Current navigation target marker ID
nav_reached  = False

# Trail
trail_x   = []
trail_y   = []
trail_z   = []
MAX_TRAIL = 100

# ─── LOAD MAP IF EXISTS ────────────────────────
def load_map():
    global marker_map
    if os.path.exists(MAP_FILE):
        with open(MAP_FILE, 'r') as f:
            marker_map = json.load(f)
        print(f"✅ Loaded map with {len(marker_map)} markers!")
    else:
        print("ℹ No existing map found. Starting fresh.")

def save_map():
    with open(MAP_FILE, 'w') as f:
        json.dump(marker_map, f, indent=2)
    print(f"💾 Map saved! {len(marker_map)} markers.")

# ─── NEAREST MARKER ────────────────────────────
def get_nearest_unvisited_marker():
    if not marker_map:
        return None

    nearest_id   = None
    nearest_dist = float('inf')

    for mid, data in marker_map.items():
        # Skip already visited in this nav session
        if data.get('visited_nav', False):
            continue

        dx = data['x'] - drone_pos[0]
        dy = data['y'] - drone_pos[1]
        dz = data['z'] - drone_pos[2]
        dist = math.sqrt(dx**2 + dy**2 + dz**2)

        if dist < nearest_dist:
            nearest_dist = dist
            nearest_id   = mid

    return nearest_id

# ─── PID ───────────────────────────────────────
POS_KP = 100.0
POS_KI = 0.0
POS_KD = 20.0

class PID:
    def __init__(self, kp, ki, kd, min_out, max_out):
        self.kp         = kp
        self.ki         = ki
        self.kd         = kd
        self.min_out    = min_out
        self.max_out    = max_out
        self.prev_error = 0.0
        self.integral   = 0.0
        self.last_time  = time.time()

    def update(self, error):
        now = time.time()
        dt  = now - self.last_time
        if dt <= 0: dt = 0.01
        self.integral  += error * dt
        self.integral   = max(-100, min(100, self.integral))
        derivative      = (error - self.prev_error) / dt
        output          = (self.kp * error +
                           self.ki * self.integral +
                           self.kd * derivative)
        self.prev_error = error
        self.last_time  = now
        return max(self.min_out, min(self.max_out, output))

    def reset(self):
        self.prev_error = 0.0
        self.integral   = 0.0
        self.last_time  = time.time()

roll_pid  = PID(POS_KP, POS_KI, POS_KD, -200, 200)
pitch_pid = PID(POS_KP, POS_KI, POS_KD, -200, 200)

# ─── THROTTLE FROM DISTANCE ────────────────────
def distance_to_throttle(distance):
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
    global manual_throttle, spinning, landing, nav_mode
    print("🛬 Landing sequence started...")
    nav_mode = False
    current  = manual_throttle
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
    global marker_x, marker_y, marker_distance
    global marker_detected, current_marker_id
    global marker_map, drone_pos

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
    prev_dist   = 0.0

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
                        x = tvec[0][0]
                        y = tvec[1][0]
                        z = tvec[2][0]

                        # Spike filter
                        if prev_dist > 0 and abs(z - prev_dist) > 0.3:
                            print(f"  ⚠ Spike ignored: {z:.2f}m")
                            continue

                        prev_dist = z

                        cv2.drawFrameAxes(frame, camera_matrix,
                                          dist_coeffs, rvec, tvec, 0.05)

                        marker_id  = ids[i][0]
                        marker_key = str(marker_id)

                        with lock:
                            marker_x          = x
                            marker_y          = y
                            marker_distance   = z
                            marker_detected   = True
                            current_marker_id = marker_id
                            drone_pos         = [x, y, z]

                        # ─── MAPPING MODE ───────
                        if mapping_mode:
                            if marker_key not in marker_map:
                                marker_map[marker_key] = {
                                    'x': round(x, 3),
                                    'y': round(y, 3),
                                    'z': round(z, 3),
                                    'visits': 1,
                                    'visited_nav': False
                                }
                                print(f"  📍 NEW marker {marker_id} mapped! "
                                      f"X:{x:.2f} Y:{y:.2f} Z:{z:.2f}")
                                save_map()
                            else:
                                # Update with average
                                old = marker_map[marker_key]
                                v   = old['visits'] + 1
                                marker_map[marker_key] = {
                                    'x': round((old['x'] * old['visits'] + x) / v, 3),
                                    'y': round((old['y'] * old['visits'] + y) / v, 3),
                                    'z': round((old['z'] * old['visits'] + z) / v, 3),
                                    'visits': v,
                                    'visited_nav': old.get('visited_nav', False)
                                }

                        auto_thr = distance_to_throttle(z)
                        top_left = (int(corner[0][0][0]),
                                    int(corner[0][0][1]))

                        if x > 0.05:
                            direction = "LEFT"
                            dir_color = (0, 165, 255)
                        elif x < -0.05:
                            direction = "RIGHT"
                            dir_color = (0, 165, 255)
                        else:
                            direction = "CENTERED"
                            dir_color = (0, 255, 0)

                        # ─── ON SCREEN ─────────
                        cv2.putText(frame, f"ID: {marker_id}",
                            (top_left[0], top_left[1] - 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame, f"Dist: {z:.3f}m",
                            (top_left[0], top_left[1] - 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                        cv2.putText(frame, f"Thr: {auto_thr}",
                            (top_left[0], top_left[1] - 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 100), 2)
                        cv2.putText(frame, f"X:{x:.2f} Y:{y:.2f}",
                            (top_left[0], top_left[1] - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        cv2.putText(frame, direction,
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, dir_color, 2)

                        # Mode display
                        if mapping_mode:
                            mode_txt   = f"🗺 MAPPING ({len(marker_map)} markers)"
                            mode_color = (0, 255, 255)
                        elif nav_mode:
                            mode_txt   = f"🤖 NAV → Marker {nav_target}"
                            mode_color = (255, 100, 0)
                        elif auto_mode:
                            mode_txt   = "AUTO HOVER"
                            mode_color = (0, 255, 0)
                        else:
                            mode_txt   = "MANUAL (A/M/N)"
                            mode_color = (0, 165, 255)

                        cv2.putText(frame, mode_txt,
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, mode_color, 2)

            else:
                with lock:
                    marker_detected   = False
                    current_marker_id = -1
                prev_dist = 0.0

                cv2.putText(frame, "No marker",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

                if mapping_mode:
                    cv2.putText(frame,
                        f"🗺 MAPPING ({len(marker_map)} markers)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)

            last_frame = frame.copy()

        if last_frame is not None:
            cv2.imshow("ArUco Drone", last_frame)
        else:
            cv2.imshow("ArUco Drone", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ─── CONTROL LOOP THREAD ───────────────────────
def control_loop():
    global manual_throttle, spinning, landing
    global auto_mode, mapping_mode, nav_mode, nav_target

    arm()
    spinning = True

    print("\n✅ Ready! Controls:")
    print("  W   - Throttle Up")
    print("  S   - Throttle Down")
    print("  A   - Toggle AUTO hover")
    print("  M   - Toggle MAPPING mode")
    print("  N   - Start NAVIGATION mode")
    print("  Q   - Stop & Disarm")
    print("  E   - Arm & Start Again")
    print("  L   - Land")
    print("  T   - End Program\n")

    while True:
        with lock:
            detected = marker_detected
            mx       = marker_x
            my       = marker_y
            mz       = marker_distance
            mid      = current_marker_id

        # ─── NAV MODE ──────────────────────────
        if nav_mode and spinning and not landing:
            if nav_target is None:
                nav_target = get_nearest_unvisited_marker()
                if nav_target is None:
                    print("  ✅ All markers visited!")
                    nav_mode = False
                else:
                    print(f"  🤖 Navigating to marker {nav_target}...")

            elif detected and str(mid) == str(nav_target):
                # Reached target marker
                auto_thr     = distance_to_throttle(mz)
                roll_adjust  = roll_pid.update(mx)
                pitch_adjust = pitch_pid.update(my)
                roll_out     = int(max(1300, min(1700, 1500 + roll_adjust)))
                pitch_out    = int(max(1300, min(1700, 1500 + pitch_adjust)))
                send_rc(roll_out, pitch_out, auto_thr)

                # Check if centered
                if abs(mx) < 0.05 and abs(my) < 0.05:
                    print(f"  ✅ Marker {nav_target} reached & centered!")
                    marker_map[str(nav_target)]['visited_nav'] = True
                    save_map()
                    time.sleep(2)  # Hover for 2 seconds
                    nav_target = get_nearest_unvisited_marker()
                    if nav_target:
                        print(f"  🤖 Next: marker {nav_target}")
                    else:
                        print("  ✅ All markers visited! Landing...")
                        nav_mode = False
                        landing  = True
                        land()
            else:
                # Searching — hold position
                send_rc(1500, 1500, manual_throttle)
                print(f"  🔍 Searching for marker {nav_target}...")

        # ─── AUTO HOVER MODE ───────────────────
        elif auto_mode and spinning and not landing:
            if detected:
                auto_thr     = distance_to_throttle(mz)
                roll_adjust  = roll_pid.update(mx)
                pitch_adjust = pitch_pid.update(my)
                roll_out     = int(max(1300, min(1700, 1500 + roll_adjust)))
                pitch_out    = int(max(1300, min(1700, 1500 + pitch_adjust)))
                send_rc(roll_out, pitch_out, auto_thr)
                print(f"  🤖 Dist:{mz:.2f}m Thr:{auto_thr} "
                      f"R:{roll_out} P:{pitch_out}")
            else:
                send_rc(1500, 1500, manual_throttle)

        # ─── MANUAL ────────────────────────────
        elif spinning and not landing:
            send_rc(1500, 1500, manual_throttle)
        elif not landing:
            send_rc(1500, 1500, 1000)

        # ─── KEYBOARD ──────────────────────────
        if msvcrt.kbhit():
            key = msvcrt.getch().decode('utf-8').lower()

            if key == 'a':
                if not nav_mode:
                    auto_mode = not auto_mode
                    if auto_mode:
                        roll_pid.reset()
                        pitch_pid.reset()
                        print("  🤖 AUTO HOVER ON")
                    else:
                        print("  🕹 MANUAL")

            elif key == 'm':
                mapping_mode = not mapping_mode
                if mapping_mode:
                    print(f"  🗺 MAPPING ON ({len(marker_map)} markers so far)")
                else:
                    save_map()
                    print(f"  🗺 MAPPING OFF — {len(marker_map)} markers saved")

            elif key == 'n':
                if not marker_map:
                    print("  ⚠ No map! Enable mapping (M) first.")
                elif not spinning:
                    print("  ⚠ Motors not running!")
                else:
                    # Reset visited flags
                    for k in marker_map:
                        marker_map[k]['visited_nav'] = False
                    nav_mode   = True
                    nav_target = None
                    auto_mode  = False
                    roll_pid.reset()
                    pitch_pid.reset()
                    print(f"  🤖 NAVIGATION MODE ON! "
                          f"{len(marker_map)} waypoints loaded")

            elif key == 'w':
                if spinning and not landing and not auto_mode and not nav_mode:
                    manual_throttle = min(manual_throttle + STEP, MAX_THROTTLE)
                    print(f"  ⬆ Throttle: {manual_throttle}")
                elif auto_mode or nav_mode:
                    print("  ⚠ In AUTO/NAV mode!")

            elif key == 's':
                if spinning and not landing and not auto_mode and not nav_mode:
                    manual_throttle = max(manual_throttle - STEP, MIN_THROTTLE)
                    print(f"  ⬇ Throttle: {manual_throttle}")
                elif auto_mode or nav_mode:
                    print("  ⚠ In AUTO/NAV mode!")

            elif key == 'q':
                if landing:
                    print("  ⚠ Landing!")
                elif spinning:
                    auto_mode    = False
                    nav_mode     = False
                    mapping_mode = False
                    spinning     = False
                    manual_throttle = 1200
                    disarm()
                else:
                    print("  ⚠ Already stopped!")

            elif key == 'e':
                if not spinning and not landing:
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
                    nav_mode  = False
                else:
                    print("  ⚠ Already running!")

            elif key == 'l':
                if spinning and not landing:
                    auto_mode    = False
                    nav_mode     = False
                    mapping_mode = False
                    landing      = True
                    spinning     = False
                    land()
                else:
                    print("  ⚠ Not flying!")

            elif key == 't':
                print("\nShutting down...")
                auto_mode    = False
                nav_mode     = False
                mapping_mode = False
                if spinning or landing:
                    land()
                else:
                    disarm()
                save_map()
                print("✅ Goodbye!")
                plt.close('all')
                break

        time.sleep(0.1)

# ─── 3D VISUALIZER (MAIN THREAD) ───────────────
def run_visualizer():
    fig = plt.figure(figsize=(8, 7))
    fig.patch.set_facecolor('#1a1a2e')
    ax  = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.cla()
        ax.set_facecolor('#1a1a2e')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(0,  3)
        ax.set_xlabel('X', color='white')
        ax.set_ylabel('Y', color='white')
        ax.set_zlabel('Z (Height)', color='white')
        ax.tick_params(colors='white')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#444')
        ax.yaxis.pane.set_edgecolor('#444')
        ax.zaxis.pane.set_edgecolor('#444')

        # Floor grid
        for g in np.arange(-2, 2.1, 0.5):
            ax.plot([g, g],  [-2, 2], [0, 0], color='#333', linewidth=0.5)
            ax.plot([-2, 2], [g, g],  [0, 0], color='#333', linewidth=0.5)

        # ─── DRAW ALL MAPPED MARKERS ───────────
        for mid, data in marker_map.items():
            mx = data['x']
            my = data['y']
            mz = data['z']
            visited = data.get('visited_nav', False)

            # Marker square
            ms = 0.12
            color = '#00ff00' if visited else 'yellow'
            ax.plot([-ms+mx,  ms+mx,  ms+mx, -ms+mx, -ms+mx],
                    [-ms+my, -ms+my,  ms+my,  ms+my, -ms+my],
                    [0, 0, 0, 0, 0],
                    color=color, linewidth=2)

            # Vertical line
            ax.plot([mx, mx], [my, my], [0, mz],
                    color=color, linewidth=1, linestyle='--', alpha=0.5)

            # Label
            status = "✓" if visited else f"#{mid}"
            ax.text(mx, my, 0.1, status,
                    color=color, fontsize=9, ha='center')

        # ─── DRAW NAV PATH ─────────────────────
        if nav_mode and nav_target and str(nav_target) in marker_map:
            tx = marker_map[str(nav_target)]['x']
            ty = marker_map[str(nav_target)]['y']
            tz = marker_map[str(nav_target)]['z']
            with lock:
                dx = drone_pos[0]
                dy = drone_pos[1]
                dz = drone_pos[2]
            ax.plot([dx, tx], [dy, ty], [dz, tz],
                    color='#ff8800', linewidth=2,
                    linestyle='--')

        # ─── TRAIL ─────────────────────────────
        with lock:
            detected = marker_detected
            dx = drone_pos[0]
            dy = drone_pos[1]
            dz = drone_pos[2]

        if detected:
            trail_x.append(dx)
            trail_y.append(dy)
            trail_z.append(dz)
            if len(trail_x) > MAX_TRAIL:
                trail_x.pop(0)
                trail_y.pop(0)
                trail_z.pop(0)

        if len(trail_x) > 1:
            ax.plot(trail_x, trail_y, trail_z,
                    color='#00aaff', linewidth=1, alpha=0.4)

        # ─── DRONE ─────────────────────────────
        if detected:
            ax.scatter([dx], [dy], [dz],
                       color='#00ff88', s=150, zorder=5)
            arm_len = 0.1
            ax.plot([dx-arm_len, dx+arm_len], [dy, dy], [dz, dz],
                    color='#00ff88', linewidth=3)
            ax.plot([dx, dx], [dy-arm_len, dy+arm_len], [dz, dz],
                    color='#00ff88', linewidth=3)
            ax.plot([dx, dx], [dy, dy], [0, dz],
                    color='#888', linewidth=1, linestyle='--')

            auto_thr = distance_to_throttle(dz)
            ax.text(dx, dy, dz + 0.15,
                    f"Drone\n{dz:.2f}m\nThr:{auto_thr}",
                    color='#00ff88', fontsize=8, ha='center')

        # ─── TITLE ─────────────────────────────
        if mapping_mode:
            mode = f"MAPPING ({len(marker_map)} markers)"
        elif nav_mode:
            mode = f"NAVIGATING → Marker {nav_target}"
        elif auto_mode:
            mode = "AUTO HOVER"
        else:
            mode = "MANUAL"

        ax.set_title(
            f"Drone Map | {mode} | "
            f"Markers: {len(marker_map)}",
            color='white', fontsize=10)

    anim = FuncAnimation(fig, update, interval=150, cache_frame_data=False)
    plt.tight_layout()
    plt.show()

# ─── INIT ──────────────────────────────────────
load_map()

# ─── START THREADS ─────────────────────────────
t1 = threading.Thread(target=aruco_thread, daemon=True)
t2 = threading.Thread(target=control_loop, daemon=True)
t1.start()
t2.start()

# ─── 3D VISUALIZER ON MAIN THREAD ──────────────
run_visualizer()