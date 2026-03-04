from pymavlink import mavutil
import time

master = mavutil.mavlink_connection('COM13', baud=57600)

print("Waiting for heartbeat...")
master.wait_heartbeat()
print(f"Connected! System: {master.target_system}")

def send_throttle(val):
    master.mav.rc_channels_override_send(
        master.target_system,
        master.target_component,
        1500, 1500, val, 1500,
        0, 0, 0, 0
    )

# ─── STEP 1: SEND MAX THROTTLE ─────────────────
print("\n⚠️  UNPLUG BATTERY NOW if not already done!")
input("Press ENTER when battery is unplugged...")

print("Sending MAX throttle (2000)...")
send_throttle(2000)
time.sleep(2)

# ─── STEP 2: PLUG IN BATTERY ───────────────────
print("\n🔋 PLUG IN BATTERY NOW!")
print("Wait for 2 beeps from ESCs...")
input("Press ENTER when you hear 2 beeps...")

# ─── STEP 3: SEND MIN THROTTLE ─────────────────
print("Sending MIN throttle (1000)...")
send_throttle(1000)

print("\nListen for:")
print("  - Beeps = number of battery cells (should be 3)")
print("  - Long beep = calibration complete ✅")
input("Press ENTER when you hear the long beep...")

# ─── STEP 4: TEST ──────────────────────────────
print("\n✅ Calibration done!")
print("Testing motors slowly...")
time.sleep(2)

for thr in range(1000, 1300, 50):
    print(f"  Throttle: {thr}")
    send_throttle(thr)
    time.sleep(1)

# Bring back to min
send_throttle(1000)
print("\n✅ All done! All 4 motors should have spun equally.")
print("If one motor was slower/faster → repeat calibration.")
