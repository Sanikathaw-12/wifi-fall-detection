import serial
import numpy as np
import re
import time
from twilio.rest import Client
from config import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, TWILIO_TO_NUMBER

PORT = "COM17"
BAUD = 115200

SPIKE_THRESHOLD = 32
STILLNESS_THRESHOLD = 16
CONFIRM_WINDOW = 6.0        # give more time for genuine movement to show up
MIN_SAMPLES_REQUIRED = 5    # need at least this many readings to trust the average
COOLDOWN_SECONDS = 20

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def send_alert():
    message = client.messages.create(
        body="⚠️ Fall detected! Please check on the person immediately.",
        from_=TWILIO_FROM_NUMBER,
        to=TWILIO_TO_NUMBER
    )
    print(f"ALERT SENT (sid: {message.sid})")

def parse_csi_line(line):
    match = re.search(r"\[(.*?)\]", line)
    if not match:
        return None
    try:
        return np.array([int(x) for x in match.group(1).split()])
    except ValueError:
        return None

def main():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    print(f"Listening on {PORT}... move around to test. Press Ctrl+C to stop.")

    buffer = []
    last_alert_time = 0
    spike_time = None
    post_spike_energies = []

    try:
        while True:
            try:
                line = ser.readline().decode(errors="ignore").strip()
            except Exception:
                continue

            if "CSI_DATA" not in line:
                continue

            csi = parse_csi_line(line)
            if csi is None:
                continue

            buffer.append(csi)
            if len(buffer) > 20:
                buffer.pop(0)
            if len(buffer) < 2:
                continue

            diffs = np.abs(np.diff(np.array(buffer[-2:]), axis=0))
            energy = diffs.mean()
            now = time.time()

            state = f"WATCHING ({len(post_spike_energies)} samples)" if spike_time else "normal"
            print(f"energy: {energy:5.1f}  state: {state}", end="\r")

            if spike_time is None:
                if energy > SPIKE_THRESHOLD:
                    spike_time = now
                    post_spike_energies = []
                    print(f"\nSpike detected (energy={energy:.1f}), watching for stillness...")
            else:
                post_spike_energies.append(energy)
                elapsed = now - spike_time

                if elapsed >= CONFIRM_WINDOW:
                    if len(post_spike_energies) < MIN_SAMPLES_REQUIRED:
                        print(f"\nNot enough data during window ({len(post_spike_energies)} samples) — inconclusive, ignoring.")
                    else:
                        avg_after = np.mean(post_spike_energies)
                        if avg_after < STILLNESS_THRESHOLD:
                            if now - last_alert_time > COOLDOWN_SECONDS:
                                print(f"\nFALL CONFIRMED (post-spike avg energy={avg_after:.1f}, {len(post_spike_energies)} samples)")
                                send_alert()
                                last_alert_time = now
                        else:
                            print(f"\nSpike NOT followed by stillness (avg={avg_after:.1f}, {len(post_spike_energies)} samples) — likely normal movement.")

                    spike_time = None
                    post_spike_energies = []

    except KeyboardInterrupt:
        print("\nStopped by user.")
        ser.close()

if __name__ == "__main__":
    main()