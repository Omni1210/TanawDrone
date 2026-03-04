from pymavlink import mavutil
import time
import msvcrt

master = mavutil.mavlink_connection('COM13', baud=57600)

print("Waiting for heartbeat...")
master.wait_heartbeat()
print(f"Connected! System: {master.target_system}")

# Disable prearm checks
print("Disabling prearm checks...")
master.mav.param_set_send(
    master.target_system,
    master.target_component,
    b'ARMING_CHECK',
    0,
    mavutil.mavlink.MAV_PARAM_TYPE_INT32
)
time.sleep(2)

# Set Stabilize mode
print("Setting Stabilize mode...")
master.mav.set_mode_send(
    master.target_system,
    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
    0
)
time.sleep(2)

# ─── FUNCTIONS ─────────────────────────────────
def send_throttle(val):
    master.mav.rc_channels_override_send(
        master.target_system,
        master.target_component,
        1500, 1500, val, 1500,
        0, 0, 0, 0
    )

def arm():
    print("▶ Arming...")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1, 0, 0, 0, 0, 0, 0
    )
    time.sleep(3)
    print("▶ Armed! Motors ready.")

def disarm():
    for _ in range(5):
        send_throttle(1000)
        time.sleep(0.05)
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        0, 0, 0, 0, 0, 0, 0
    )
    print("⏹ Disarmed! Motors completely stopped.")

# ─── SETTINGS ──────────────────────────────────
throttle = 1200
MIN_THROTTLE = 1100
MAX_THROTTLE = 2000
STEP = 50
spinning = False

# ─── INITIAL ARM ───────────────────────────────
arm()
spinning = True

# ─── MAIN LOOP ─────────────────────────────────
print("\n✅ Ready! Controls:")
print("  W - Speed Up")
print("  S - Speed Down")
print("  Q - Stop & Disarm")
print("  E - Arm & Start Again")
print("  T - End Program\n")

while True:
    # Keep sending throttle
    if spinning:
        send_throttle(throttle)
    else:
        send_throttle(1000)

    if msvcrt.kbhit():
        key = msvcrt.getch().decode('utf-8').lower()

        if key == 'w':
            if spinning:
                throttle = min(throttle + STEP, MAX_THROTTLE)
                print(f"  ⬆ Throttle: {throttle}")
            else:
                print("  ⚠ Motors not running! Press E first.")

        elif key == 's':
            if spinning:
                throttle = max(throttle - STEP, MIN_THROTTLE)
                print(f"  ⬇ Throttle: {throttle}")
            else:
                print("  ⚠ Motors not running! Press E first.")

        elif key == 'q':
            if spinning:
                spinning = False
                throttle = 1200  # Reset for next start
                disarm()
            else:
                print("  ⚠ Motors already stopped!")

        elif key == 'e':
            if not spinning:
                arm()
                spinning = True
                print(f"  ▶ Starting at throttle: {throttle}")
            else:
                print("  ⚠ Motors already running!")

        elif key == 't':
            print("\nShutting down...")
            spinning = False
            disarm()
            print("✅ Goodbye!")
            break

    time.sleep(0.1)