import cv2
import numpy as np

cap = cv2.VideoCapture(2)

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

MARKER_SIZE = 0.10  # Change to your actual marker size in meters

# ─── NEW: Define marker points ──────────────────
obj_points = np.array([
    [-MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
    [ MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
    [ MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
    [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0]
], dtype=np.float32)

print("ArUco Detection Started!")
print("Press Q to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i, corner in enumerate(corners):
            # ─── NEW: Use solvePnP instead ──────
            img_points = corner[0]
            success, rvec, tvec = cv2.solvePnP(
                obj_points, img_points, camera_matrix, dist_coeffs
            )

            if success:
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

                distance = np.linalg.norm(tvec)
                x = tvec[0][0]
                y = tvec[1][0]
                z = tvec[2][0]

                marker_id = ids[i][0]

                cv2.putText(frame,
                    f"ID: {marker_id}",
                    (int(corner[0][0][0]), int(corner[0][0][1]) - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.putText(frame,
                    f"Dist: {distance:.2f}m",
                    (int(corner[0][0][0]), int(corner[0][0][1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                print(f"Marker ID: {marker_id}")
                print(f"  Distance: {distance:.3f} m")
                print(f"  X: {x:.3f}  Y: {y:.3f}  Z: {z:.3f}")
                print()

    cv2.imshow("ArUco Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()