import cv2
import numpy as np
import math

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

focal_length = 445
center = (640 / 2, 480 / 2)

camera_matrix = np.array([
    [focal_length, 0,            center[0]],
    [0,            focal_length, center[1]],
    [0,            0,            1         ]
], dtype=np.float32)

dist_coeffs = np.zeros((4, 1))

MARKER_SIZE = 0.10

obj_points = np.array([
    [-MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
    [ MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
    [ MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
    [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0]
], dtype=np.float32)

def draw_diagram(frame, distance, yaw, x, y):
    panel_x = frame.shape[1] - 220
    panel_y = frame.shape[0] - 220
    panel_w = 210
    panel_h = 210

    cv2.rectangle(frame,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (30, 30, 30), -1)
    cv2.rectangle(frame,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (100, 100, 100), 1)

    cv2.putText(frame, "TOP VIEW",
        (panel_x + 60, panel_y + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cam_x = panel_x + panel_w // 2
    cam_y = panel_y + 40

    cv2.rectangle(frame,
        (cam_x - 12, cam_y - 8),
        (cam_x + 12, cam_y + 8),
        (0, 255, 0), 2)
    cv2.putText(frame, "CAM",
        (cam_x - 14, cam_y - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    scale = 80

    marker_draw_x = int(cam_x + x * scale)
    marker_draw_y = int(cam_y + distance * scale)
    marker_draw_x = max(panel_x + 15, min(panel_x + panel_w - 15, marker_draw_x))
    marker_draw_y = max(panel_y + 15, min(panel_y + panel_h - 15, marker_draw_y))

    cv2.line(frame, (cam_x, cam_y), (marker_draw_x, marker_draw_y), (0, 255, 255), 1)
    cv2.line(frame,
        (cam_x, cam_y),
        (cam_x, min(cam_y + int(distance * scale), panel_y + panel_h - 15)),
        (80, 80, 80), 1, cv2.LINE_AA)

    angle_rad = math.atan2(x, distance)
    angle_deg = math.degrees(angle_rad)
    cv2.ellipse(frame, (cam_x, cam_y), (25, 25), -90, 0, angle_deg, (255, 165, 0), 1)

    draw_marker_icon(frame, marker_draw_x, marker_draw_y, yaw)

    cv2.putText(frame, f"{angle_deg:.1f}deg",
        (cam_x + 5, cam_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 165, 0), 1)

    mid_x = (cam_x + marker_draw_x) // 2
    mid_y = (cam_y + marker_draw_y) // 2
    cv2.putText(frame, f"{distance:.2f}m",
        (mid_x + 5, mid_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

def draw_marker_icon(frame, cx, cy, yaw):
    size = 10
    angle = math.radians(yaw)
    pts = np.array([
        [-size, -size],
        [ size, -size],
        [ size,  size],
        [-size,  size]
    ], dtype=np.float32)

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rotated = []
    for p in pts:
        rx = int(cx + p[0] * cos_a - p[1] * sin_a)
        ry = int(cy + p[0] * sin_a + p[1] * cos_a)
        rotated.append([rx, ry])

    rotated = np.array(rotated, dtype=np.int32)
    cv2.polylines(frame, [rotated], True, (0, 100, 255), 2)
    cv2.putText(frame, "M",
        (cx - 5, cy + 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 100, 255), 1)

print("ArUco Detection Started!")
print("Press Q to quit\n")

frame_count = 0
last_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error!")
        break

    frame_count += 1

    # ─── PROCESS EVERY 2ND FRAME ───────────────
    if frame_count % 2 == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for i, corner in enumerate(corners):
                img_points = corner[0]
                success, rvec, tvec = cv2.solvePnP(
                    obj_points, img_points, camera_matrix, dist_coeffs
                )

                if success:
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

                    distance = tvec[2][0]
                    x = tvec[0][0]
                    y = tvec[1][0]
                    z = tvec[2][0]

                    rmat, _ = cv2.Rodrigues(rvec)
                    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                    pitch = angles[0]
                    yaw   = angles[1]
                    roll  = angles[2]

                    h_angle = math.degrees(math.atan2(x, distance))

                    marker_id = ids[i][0]
                    top_left = (int(corner[0][0][0]), int(corner[0][0][1]))

                    if x > 0.05:
                        direction = "Marker LEFT"
                        dir_color = (0, 165, 255)
                    elif x < -0.05:
                        direction = "Marker RIGHT"
                        dir_color = (0, 165, 255)
                    else:
                        direction = "Marker CENTERED"
                        dir_color = (0, 255, 0)

                    cv2.putText(frame, f"ID: {marker_id}",
                        (top_left[0], top_left[1] - 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"Dist: {distance:.3f} m",
                        (top_left[0], top_left[1] - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(frame, f"Angle: {h_angle:.1f} deg",
                        (top_left[0], top_left[1] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                    cv2.putText(frame, f"X:{x:.2f} Y:{y:.2f} Z:{z:.2f}",
                        (top_left[0], top_left[1] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    cv2.putText(frame, f"P:{pitch:.1f} Y:{yaw:.1f} R:{roll:.1f}",
                        (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)
                    cv2.putText(frame, direction,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, dir_color, 2)

                    draw_diagram(frame, distance, yaw, x, y)

                    print(f"Marker ID : {marker_id}")
                    print(f"  Distance : {distance:.3f} m")
                    print(f"  Angle    : {h_angle:.1f} deg")
                    print(f"  X: {x:.3f}  Y: {y:.3f}  Z: {z:.3f}")
                    print(f"  Pitch: {pitch:.1f}  Yaw: {yaw:.1f}  Roll: {roll:.1f}")
                    print(f"  {direction}")
                    print()

        else:
            cv2.putText(frame, "No marker detected",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ─── SAVE LAST PROCESSED FRAME ─────────
        last_frame = frame.copy()

    # ─── ALWAYS SHOW LAST PROCESSED FRAME ──────
    if last_frame is not None:
        cv2.imshow("ArUco Detection", last_frame)
    else:
        cv2.imshow("ArUco Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()