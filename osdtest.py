import struct
import time
import threading
import serial
from pymavlink import mavutil

# ── Config ──────────────────────────────────────────────────────────────
SERIAL_MAVLINK = '/dev/serial0'   # MAVLink telemetry UART
SERIAL_MSP     = '/dev/serial1'   # MSP DisplayPort UART
BAUD           = 115200

FRAME_W, FRAME_H   = 640, 480
OSD_COLS, OSD_ROWS = 30, 16

CH_X        = 12     # Aim X knob
CH_Y        = 13    # Aim Y knob
CH_LOCK     = 8     # Lock switch
LOCK_THRESH = 1700

# ── State ───────────────────────────────────────────────────────────────
state = {
    'aim_x': FRAME_W // 2,
    'aim_y': FRAME_H // 2,

    'last_col': -1,
    'last_row': -1,

    'locked': False,
}

# ── MSP DisplayPort Commands ────────────────────────────────────────────
MSP_DP_CLEAR = 2
MSP_DP_WRITE = 3
MSP_DP_DRAW  = 4

# ── MSP Frame Builder ───────────────────────────────────────────────────
def msp_frame(payload: bytes) -> bytes:
    length = len(payload)

    # MSPv2 header
    header = struct.pack('<BBHH',
        0x58,      # flags
        0,         # function
        0x0300,    # DisplayPort
        length
    )

    crc = 0
    for b in header[1:] + payload:
        crc ^= b

    return b'$X<' + header + payload + bytes([crc])

# ── DisplayPort Helpers ─────────────────────────────────────────────────
def dp_write(col, row, char):
    col = max(0, min(col, OSD_COLS - 1))
    row = max(0, min(row, OSD_ROWS - 1))

    payload = struct.pack(
        'BBBB',
        MSP_DP_WRITE,
        row,
        col,
        ord(char)
    )

    return msp_frame(payload)

def dp_draw():
    return msp_frame(struct.pack('B', MSP_DP_DRAW))

# ── Coordinate Conversion ───────────────────────────────────────────────
def px_to_osd(px, py):
    col = int((px / FRAME_W) * OSD_COLS)
    row = int((py / FRAME_H) * OSD_ROWS)

    col = max(0, min(col, OSD_COLS - 1))
    row = max(0, min(row, OSD_ROWS - 1))

    return col, row

# ── PWM Mapping ─────────────────────────────────────────────────────────
def pwm_to_px(pwm, axis_max):
    pwm = max(1000, min(2000, pwm))
    return int(((pwm - 1000) / 1000.0) * axis_max)

# ── OSD Character Renderer ──────────────────────────────────────────────
def draw_marker(msp_serial, px, py, char):
    col, row = px_to_osd(px, py)

    frames = []

    # erase previous marker
    if state['last_col'] >= 0:
        frames.append(
            dp_write(
                state['last_col'],
                state['last_row'],
                ' '
            )
        )

    # draw new marker
    frames.append(dp_write(col, row, char))

    # update display
    frames.append(dp_draw())

    # send all at once
    packet = b''.join(frames)
    msp_serial.write(packet)

    # save previous location
    state['last_col'] = col
    state['last_row'] = row

# ── MAVLink RC Reader ───────────────────────────────────────────────────
def mavlink_loop(mav, msp_serial):

    print("MAVLink loop started")

    while True:

        msg = mav.recv_match(
            type='RC_CHANNELS',
            blocking=True,
            timeout=0.1
        )

        if msg is None:
            continue

        ch_x_pwm = getattr(msg, f'chan{CH_X}_raw', 1500)
        ch_y_pwm = getattr(msg, f'chan{CH_Y}_raw', 1500)
        ch_lock  = getattr(msg, f'chan{CH_LOCK}_raw', 1000)

        # move aim point with knobs
        state['aim_x'] = pwm_to_px(ch_x_pwm, FRAME_W)
        state['aim_y'] = pwm_to_px(ch_y_pwm, FRAME_H)

        # lock switch
        state['locked'] = ch_lock > LOCK_THRESH

        # unlocked = "+"
        # locked   = "X"
        marker = 'X' if state['locked'] else '+'

        draw_marker(
            msp_serial,
            state['aim_x'],
            state['aim_y'],
            marker
        )

        time.sleep(0.03)

# ── Tracking Callback ───────────────────────────────────────────────────
def on_track_update(msp_serial, cx, cy):

    if not state['locked']:
        return

    draw_marker(
        msp_serial,
        cx,
        cy,
        'X'
    )

# ── Main ────────────────────────────────────────────────────────────────
msp_ser = serial.Serial(SERIAL_MSP, BAUD)

mav = mavutil.mavlink_connection(
    SERIAL_MAVLINK,
    baud=BAUD
)

mav.wait_heartbeat()

print("Flight controller heartbeat received")

t = threading.Thread(
    target=mavlink_loop,
    args=(mav, msp_ser),
    daemon=True
)

t.start()

# Your CV loop goes here
while True:
    time.sleep(1)
