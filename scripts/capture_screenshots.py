import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "docs" / "screenshots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

PAGES = [
    ("dashboard", "http://localhost:5173/"),
    ("predictive_maintenance", "http://localhost:5173/prediction"),
    ("maintenance_requests", "http://localhost:5173/maintenance"),
    ("block_planning", "http://localhost:5173/planning"),
    ("model_insights", "http://localhost:5173/insights"),
    ("privacy_policy", "http://localhost:5173/privacy"),
    ("not_found", "http://localhost:5173/non-existent-route"),
]

VIEWPORTS = [
    ("desktop", 1440, 950),
    ("mobile", 390, 844),
]

def capture():
    print(f"Capturing screenshots to {OUTPUT_DIR}...")
    for name, url in PAGES:
        for vp_name, width, height in VIEWPORTS:
            out_file = OUTPUT_DIR / f"{name}_{vp_name}.png"
            cmd = [
                CHROME_BIN,
                "--headless",
                "--disable-gpu",
                f"--window-size={width},{height}",
                "--virtual-time-budget=4000",
                f"--screenshot={out_file}",
                url,
            ]
            print(f" -> {name} [{vp_name}] ({width}x{height}) ...")
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"    Saved: {out_file.name} ({out_file.stat().st_size} bytes)")
            except Exception as e:
                print(f"    Failed {name} [{vp_name}]: {e}")

    print("All screenshots captured successfully!")

if __name__ == "__main__":
    capture()
