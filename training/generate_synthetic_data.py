"""
Generate Synthetic Demo Data - creates short test video and images with simulated billets.
Embeds both Billet ID and Heat Number on markings and 2D industrial tags.

Usage:
    python training/generate_synthetic_data.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np


def _make_qr_patch(text: str, target_size: int = 70) -> np.ndarray:
    """Generate a clean BGR QR code patch for the given text."""
    try:
        import qrcode
        qr = qrcode.QRCode(box_size=2, border=1)
        qr.add_data(text)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return cv2.resize(bgr, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    except Exception:
        # Fallback high-contrast checker pattern
        patch = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        patch[:] = (240, 240, 240)
        cv2.rectangle(patch, (5, 5), (target_size - 5, target_size - 5), (20, 20, 20), 2)
        cv2.putText(patch, "TAG", (10, target_size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return patch


def generate_demo_video(output_path: str = None, num_frames: int = 120, fps: int = 15):
    """Generate a synthetic demo video with moving billet-like objects."""
    if output_path is None:
        output_path = str(ROOT / "data" / "demo" / "demo_video.mp4")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"ERROR: Cannot create video at {output_path}")
        return

    print(f"Generating {num_frames} frames at {fps} FPS ...")

    billet_x = -400
    speed = 8

    billet_id = "BLT-1042"
    heat_number = "HT-88215"
    qr_patch = _make_qr_patch(f"BILLET:{billet_id}|HEAT:{heat_number}", 72)
    qrh, qrw = qr_patch.shape[:2]

    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Factory background
        frame[:] = (30, 28, 25)
        noise = np.random.randint(0, 10, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)

        # Conveyor lines
        cv2.line(frame, (0, 480), (width, 480), (50, 48, 45), 2)
        cv2.line(frame, (0, 240), (width, 240), (50, 48, 45), 2)

        # Moving billet
        bx = billet_x + i * speed
        by1, by2 = 280, 440
        bx1 = bx
        bx2 = bx + 600

        if bx2 > 0 and bx1 < width:
            dx1 = max(0, bx1)
            dx2 = min(width, bx2)

            # Billet body (warm steel colour)
            temp = max(0, min(1, (i / num_frames)))
            r = int(80 + 60 * (1 - temp))
            g = int(70 + 20 * (1 - temp))
            b = int(60)
            cv2.rectangle(frame, (dx1, by1), (dx2, by2), (b, g, r), -1)
            cv2.rectangle(frame, (dx1, by1), (dx2, by2), (b + 30, g + 30, r + 30), 2)

            # Surface texture
            for _ in range(80):
                sx = np.random.randint(max(dx1 + 2, 0), max(dx2 - 2, 3))
                sy = np.random.randint(by1 + 2, by2 - 2)
                cv2.circle(frame, (sx, sy), 1, (b + 15, g + 10, r + 10), -1)

            # Marking area stamped with Billet ID, Heat Number, and 2D DataMatrix/QR tag
            mark_x = bx + 220
            if 0 < mark_x < width - 360:
                plate_w = 340
                cv2.rectangle(frame, (mark_x, 305), (mark_x + plate_w, 405), (60, 55, 50), -1)
                cv2.rectangle(frame, (mark_x, 305), (mark_x + plate_w, 405), (150, 145, 140), 2)

                # Text markings
                cv2.putText(frame, f"BILLET: {billet_id}", (mark_x + 15, 345),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (240, 235, 230), 2)
                cv2.putText(frame, f"HEAT: {heat_number}", (mark_x + 15, 385),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (240, 235, 230), 2)

                # Embed QR Tag
                qx = mark_x + plate_w - qrw - 10
                qy = 318
                if 0 <= qx < width - qrw and 0 <= qy < height - qrh:
                    frame[qy:qy + qrh, qx:qx + qrw] = qr_patch

            # Simulated defect (appears on some frames)
            if 30 < i < 80:
                defect_x = bx + 160
                if 0 < defect_x < width:
                    cv2.ellipse(frame, (defect_x, 360), (25, 12), 20, 0, 360, (20, 15, 10), -1)

        # HUD overlay
        cv2.putText(frame, f"DEMO | Frame {i + 1}/{num_frames}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
        cv2.putText(frame, "SYNTHETIC FACTORY TEST VIDEO", (10, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        out.write(frame)

    out.release()
    duration = num_frames / fps
    print(f"[OK] Demo video saved: {output_path}")
    print(f"   {num_frames} frames, {fps} FPS, {duration:.1f}s duration")
    return output_path


def generate_demo_images(output_dir: str = None, count: int = 5):
    """Generate individual synthetic billet images for testing."""
    if output_dir is None:
        output_dir = str(ROOT / "data" / "demo")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for idx in range(count):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:] = (35, 32, 28)
        noise = np.random.randint(0, 12, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)

        # Billet position
        bx = np.random.randint(100, 250)
        by = np.random.randint(200, 280)
        bw = np.random.randint(650, 850)
        bh = np.random.randint(140, 220)

        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (85, 80, 75), -1)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (115, 110, 105), 2)

        # Unique Billet ID & Heat Number
        billet_id = f"BLT-{1000 + idx}"
        heat_number = f"HT-{58210 + idx}"

        # Stamped marking plate
        plate_w, plate_h = 420, 115
        px1, py1 = bx + 40, by + 30
        px2, py2 = px1 + plate_w, py1 + plate_h
        cv2.rectangle(frame, (px1, py1), (px2, py2), (60, 55, 50), -1)
        cv2.rectangle(frame, (px1, py1), (px2, py2), (150, 145, 140), 2)

        # Text
        cv2.putText(frame, f"BILLET: {billet_id}", (px1 + 15, py1 + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (240, 235, 230), 2)
        cv2.putText(frame, f"HEAT: {heat_number}", (px1 + 15, py1 + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (240, 235, 230), 2)

        # QR Tag
        qr_patch = _make_qr_patch(f"BILLET:{billet_id}|HEAT:{heat_number}", 80)
        qrh, qrw = qr_patch.shape[:2]
        frame[py1 + 18:py1 + 18 + qrh, px2 - qrw - 15:px2 - 15] = qr_patch

        # Scale bar
        cv2.line(frame, (bx, by + bh + 40), (bx + 200, by + bh + 40), (200, 200, 200), 2)
        cv2.putText(frame, "200px (20.0 cm)", (bx + 30, by + bh + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.putText(frame, "DEMO - Synthetic Billet Inspection", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

        path = Path(output_dir) / f"demo_billet_{idx + 1:03d}.jpg"
        cv2.imwrite(str(path), frame)
        print(f"  Saved: {path}")

    print(f"[OK] Generated {count} demo images in {output_dir}")


if __name__ == "__main__":
    generate_demo_video()
    generate_demo_images()
