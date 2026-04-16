import os
import sound_util


def collect_data(label, count):
    save_path = f"data/{label}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(count):
        filename = f"{save_path}/{label}_{i}.wav"
        print(f"Recording {label} {i + 1}/{count}...")
        sound_util.record_sample(filename, duration=1.0)
        print("Saved")


if __name__ == "__main__":
    target = input("Label (clap/whistle/harmonica/noise/silence): ")
    amount = int(input("Quantity: "))
    collect_data(target, amount)