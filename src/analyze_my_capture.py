import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

columns = ["type","role","mac","rssi","rate","sig_mode","mcs","bandwidth",
           "smoothing","not_sounding","aggregation","stbc","fec_coding","sgi",
           "noise_floor","ampdu_cnt","channel","secondary_channel",
           "local_timestamp","ant","sig_len","rx_state","real_time_set",
           "real_timestamp","len","CSI_DATA"]

def load_capture(filename):
    df = pd.read_csv(
        f"C:\\Users\\ASUS\\Desktop\\Projects\\{filename}",
        header=None,
        names=columns
    )
    df = df[df["type"] == "CSI_DATA"].reset_index(drop=True)
    return df

still = load_capture("still.csv")
walk = load_capture("walk.csv")
fall = load_capture("fall.csv")
standup = load_capture("standup.csv")

print("still:", len(still), "rows")
print("walk:", len(walk), "rows")
print("fall:", len(fall), "rows")
print("standup:", len(standup), "rows")

# Plot RSSI for all four activities side by side
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False)

axes[0].plot(still["rssi"].values, color='green')
axes[0].set_title("Still")
axes[0].set_ylabel("RSSI")

axes[1].plot(walk["rssi"].values, color='blue')
axes[1].set_title("Walk")
axes[1].set_ylabel("RSSI")

axes[2].plot(fall["rssi"].values, color='red')
axes[2].set_title("Fall")
axes[2].set_ylabel("RSSI")

axes[3].plot(standup["rssi"].values, color='orange')
axes[3].set_title("Standup / Normal movement")
axes[3].set_ylabel("RSSI")
axes[3].set_xlabel("Packet index (time)")

plt.tight_layout()
plt.savefig("../results/my_activities_comparison.png")
plt.show()


def parse_csi(row):
    try:
        cleaned = row.strip("[]")
        return np.array([int(x) for x in cleaned.split()])
    except (ValueError, AttributeError):
        return None

def get_csi_matrix(df):
    parsed = df["CSI_DATA"].apply(parse_csi)
    parsed = parsed.dropna()  # drop rows that failed to parse
    lengths = parsed.apply(len)
    common_len = lengths.mode()[0]
    matrix = np.stack(parsed[parsed.apply(len) == common_len].values)
    return matrix

still_matrix = get_csi_matrix(still)
walk_matrix = get_csi_matrix(walk)
fall_matrix = get_csi_matrix(fall)
standup_matrix = get_csi_matrix(standup)

print("Still CSI matrix:", still_matrix.shape)
print("Walk CSI matrix:", walk_matrix.shape)
print("Fall CSI matrix:", fall_matrix.shape)
print("Standup CSI matrix:", standup_matrix.shape)

def movement_energy(matrix):
    diffs = np.abs(np.diff(matrix, axis=0))
    return diffs.mean(axis=1)

still_energy = movement_energy(still_matrix)
walk_energy = movement_energy(walk_matrix)
fall_energy = movement_energy(fall_matrix)
standup_energy = movement_energy(standup_matrix)

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharey=True)
axes[0].plot(still_energy, color='green'); axes[0].set_title(f"Still (mean energy: {still_energy.mean():.1f})")
axes[1].plot(walk_energy, color='blue'); axes[1].set_title(f"Walk (mean energy: {walk_energy.mean():.1f})")
axes[2].plot(fall_energy, color='red'); axes[2].set_title(f"Fall (mean energy: {fall_energy.mean():.1f})")
axes[3].plot(standup_energy, color='orange'); axes[3].set_title(f"Standup (mean energy: {standup_energy.mean():.1f})")
axes[3].set_xlabel("Packet index (time)")

plt.tight_layout()
plt.savefig("../results/my_csi_energy_comparison.png")
plt.show()