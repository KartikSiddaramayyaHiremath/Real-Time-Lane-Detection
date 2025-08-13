import cv2
import numpy as np
import os

# Polygon & line drawing parameters
ROI_TOP_RATIO = 0.60
ROI_SIDE_INSET = 0.10
ALPHA_FILL = 0.35
LINE_COLOR = (0, 100, 0)
LINE_THICKNESS = 8

# Memory for last good lines
last_left_mb = None
last_right_mb = None

def region_of_interest_mask(shape):
    h, w = shape[:2]
    polygon = np.array([[
        (int(w * ROI_SIDE_INSET), h),
        (int(w * 0.45), int(h * ROI_TOP_RATIO)),
        (int(w * 0.55), int(h * ROI_TOP_RATIO)),
        (int(w * (1 - ROI_SIDE_INSET)), h)
    ]], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, polygon, 255)
    return mask

def average_slope_intercept(points):
    left_pts, right_pts = [], []
    for x1, y1, x2, y2 in points:
        if x2 == x1:
            continue
        m = (y2 - y1) / (x2 - x1)
        if abs(m) < 0.2:
            continue
        if m < 0:
            left_pts.extend([(x1, y1), (x2, y2)])
        else:
            right_pts.extend([(x1, y1), (x2, y2)])

    left_line, right_line = None, None
    if len(left_pts) >= 2:
        xs = np.array([p[0] for p in left_pts])
        ys = np.array([p[1] for p in left_pts])
        left_line = tuple(np.polyfit(xs, ys, 1))
    if len(right_pts) >= 2:
        xs = np.array([p[0] for p in right_pts])
        ys = np.array([p[1] for p in right_pts])
        right_line = tuple(np.polyfit(xs, ys, 1))

    return left_line, right_line

def line_points_from_mb(mb, y_bottom, y_top, w):
    if mb is None:
        return None
    m, b = mb
    if abs(m) < 1e-6:
        return None
    x_bottom = int((y_bottom - b) / m)
    x_top = int((y_top - b) / m)
    x_bottom = max(0, min(w - 1, x_bottom))
    x_top = max(0, min(w - 1, x_top))
    return (x_bottom, y_bottom), (x_top, y_top)

def detect_lanes(frame):
    global last_left_mb, last_right_mb
    h, w = frame.shape[:2]

    # Color filter for yellow & white
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    yellow_mask = cv2.inRange(hls, np.array([15, 30, 115]), np.array([35, 204, 255]))
    white_mask = cv2.inRange(hls, np.array([0, 200, 0]), np.array([255, 255, 255]))
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    filtered = cv2.bitwise_and(frame, frame, mask=mask)

    # Morphology
    morph = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Edges + ROI
    gray = cv2.cvtColor(morph, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    roi_mask = region_of_interest_mask(edges.shape)
    edges_roi = cv2.bitwise_and(edges, roi_mask)

    # Hough transform
    lines = cv2.HoughLinesP(edges_roi, 1, np.pi / 180, 30, minLineLength=40, maxLineGap=200)

    left_mb, right_mb = None, None
    if lines is not None:
        segs = [tuple(l) for l in lines[:, 0]]
        left_mb, right_mb = average_slope_intercept(segs)

    # Keep last detected if missing
    if left_mb is None:
        left_mb = last_left_mb
    else:
        last_left_mb = left_mb
    if right_mb is None:
        right_mb = last_right_mb
    else:
        last_right_mb = right_mb

    result = frame.copy()
    y_bottom = h
    y_top = int(h * ROI_TOP_RATIO)
    left_pts = line_points_from_mb(left_mb, y_bottom, y_top, w)
    right_pts = line_points_from_mb(right_mb, y_bottom, y_top, w)

    if left_pts and right_pts:
        poly = np.array([left_pts[0], left_pts[1], right_pts[1], right_pts[0]], dtype=np.int32)
        overlay = result.copy()
        cv2.fillPoly(overlay, [poly], (0, 255, 0))
        result = cv2.addWeighted(overlay, ALPHA_FILL, result, 1 - ALPHA_FILL, 0)
        cv2.line(result, left_pts[0], left_pts[1], LINE_COLOR, LINE_THICKNESS)
        cv2.line(result, right_pts[0], right_pts[1], LINE_COLOR, LINE_THICKNESS)

    return result

def process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for f in os.listdir(input_folder):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(input_folder, f)
            img = cv2.imread(img_path)
            if img is None:
                continue
            result = detect_lanes(img)

            # Combine original & processed side-by-side
            combined = np.hstack((img, result))

            # Save combined image
            out_path = os.path.join(output_folder, f"{os.path.splitext(f)[0]}_lanes.jpg")
            cv2.imwrite(out_path, combined)

            # Display combined image
            cv2.imshow("Original (Left)  |  Lane Detection (Right)", combined)
            if cv2.waitKey(500) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    images_in = r"E:\lane detection\images"
    images_out = r"E:\lane detection\output_images123"
    process_images(images_in, images_out)
