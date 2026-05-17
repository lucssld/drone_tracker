from pymavlink import mavutil
import time
import math

# =====================================================
# CONFIG
# =====================================================

SERIAL_PORT = "/dev/serial0"
BAUD = 115200

MOVE_DISTANCE_METERS = 5

GCS_CHANNEL = 6
GCS_THRESHOLD = 1700

# =====================================================
# CONNECT
# =====================================================

print("Connecting...")

master = mavutil.mavlink_connection(
    SERIAL_PORT,
    baud=BAUD
)

master.wait_heartbeat()

print("Heartbeat received")

# =====================================================
# WAIT FOR GPS
# =====================================================

print("Waiting for GPS data...")

while True:

    msg = master.recv_match(
        type='GLOBAL_POSITION_INT',
        blocking=True
    )

    if msg:

        current_lat = msg.lat / 1e7
        current_lon = msg.lon / 1e7
        current_alt = msg.relative_alt / 1000.0
        current_heading = msg.hdg / 100.0

        break

print(f"\nCurrent Position:")
print(f"Lat: {current_lat}")
print(f"Lon: {current_lon}")
print(f"Heading: {current_heading}")

# =====================================================
# CALCULATE TARGET
# =====================================================

north = MOVE_DISTANCE_METERS
east = 0

dlat = north / 111320.0

dlon = east / (
    111320.0 *
    math.cos(math.radians(current_lat))
)

target_lat = current_lat + dlat
target_lon = current_lon + dlon

print("\nTarget Position:")
print(f"{target_lat}, {target_lon}")

# =====================================================
# WAIT FOR CHANNEL 6
# =====================================================

print(
    f"\nWaiting for CH{GCS_CHANNEL} "
    f"> {GCS_THRESHOLD}..."
)

while True:

    msg = master.recv_match(
        type='RC_CHANNELS',
        blocking=True
    )

    if not msg:
        continue

    channel_value = getattr(
        msg,
        f'chan{GCS_CHANNEL}_raw',
        0
    )

    print(
        f"CH{GCS_CHANNEL}: "
        f"{channel_value}"
    )

    if channel_value > GCS_THRESHOLD:

        print(
            "\nGCS NAV SWITCH ACTIVE"
        )

        break

# =====================================================
# SEND TARGET
# =====================================================

print("\nSending target position...")

master.mav.set_position_target_global_int_send(

    0,

    master.target_system,
    master.target_component,

    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,

    int(0b110111111000),

    int(target_lat * 1e7),
    int(target_lon * 1e7),

    current_alt,

    0,
    0,
    0,

    0,
    0,
    0,

    0,
    0
)

print("Target command sent.")

# =====================================================
# MONITOR POSITION
# =====================================================

while True:

    msg = master.recv_match(
        type='GLOBAL_POSITION_INT',
        blocking=True
    )

    lat = msg.lat / 1e7
    lon = msg.lon / 1e7

    print(
        f"Current: "
        f"{lat:.7f}, "
        f"{lon:.7f}"
    )

    time.sleep(0.5)
