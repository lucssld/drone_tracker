from pymavlink import mavutil
import math

# =========================
# CONFIG
# =========================

SERIAL_PORT = "/dev/serial0"
BAUD = 115200

# =========================
# CONNECT
# =========================

print("Connecting to flight controller...")

master = mavutil.mavlink_connection(
    SERIAL_PORT,
    baud=BAUD
)

master.wait_heartbeat()

print("Heartbeat received!")
print(f"System ID: {master.target_system}")
print(f"Component ID: {master.target_component}")

# =========================
# MAIN LOOP
# =========================

while True:

    msg = master.recv_match(
        type=['GPS_RAW_INT', 'GLOBAL_POSITION_INT'],
        blocking=True
    )

    if not msg:
        continue

    msg_type = msg.get_type()

    # -------------------------
    # GPS DATA
    # -------------------------
    if msg_type == "GPS_RAW_INT":

        lat = msg.lat / 1e7
        lon = msg.lon / 1e7
        alt = msg.alt / 1000.0

        sats = msg.satellites_visible

        print("\n===== GPS =====")
        print(f"Lat: {lat:.7f}")
        print(f"Lon: {lon:.7f}")
        print(f"Alt: {alt:.1f} m")
        print(f"Satellites: {sats}")

    # -------------------------
    # HEADING + SPEED
    # -------------------------
    elif msg_type == "GLOBAL_POSITION_INT":

        heading = msg.hdg / 100.0
        ground_speed = math.sqrt(msg.vx**2 + msg.vy**2) / 100.0

        print("\n===== NAV =====")
        print(f"Heading: {heading:.1f}°")
        print(f"Ground Speed: {ground_speed:.2f} m/s")
