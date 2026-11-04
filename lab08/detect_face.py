import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
import warnings
from scipy.ndimage import convolve, gaussian_filter

MODEL_PATH = "version-RFB-320.onnx"

warnings.filterwarnings("ignore", category=UserWarning)

sess_options = ort.SessionOptions()
sess_options.log_severity_level = 3  

# 3. Use only available providers to prevent the provider fallback warning
desired_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
available_providers = [p for p in desired_providers if p in ort.get_available_providers()]

# face detection
def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    scores = box_scores[:, 4]
    boxes = box_scores[:, :4]

    order = np.argsort(scores)[::-1]
    if candidate_size > 0:
        order = order[:candidate_size]

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    picked = []

    while order.size > 0:
        current = order[0]
        picked.append(current)

        if top_k > 0 and len(picked) >= top_k:
            break

        xx1 = np.maximum(boxes[current, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[current, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[current, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[current, 3], boxes[order[1:], 3])

        width = np.maximum(0, xx2 - xx1)
        height = np.maximum(0, yy2 - yy1)
        intersection = width * height
        union = areas[current] + areas[order[1:]] - intersection
        iou = intersection / union

        order = order[1:][iou <= iou_threshold]

    return box_scores[picked]

def detect_faces(image_path, output_path="output_detected.png", conf_threshold=0.7):
    image = Image.open(image_path).convert("RGB")
    original_width, original_height = image.size
    resized = image.resize((320, 240))
    data = np.asarray(resized).astype(np.float32)
    data = (data - 127.0) / 128.0
    data = np.transpose(data, (2, 0, 1))
    data = np.expand_dims(data, axis=0)

    session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=sess_options,
        providers=available_providers
    )
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: data})
    confidences = outputs[0][0]
    locations = outputs[1][0]

    min_boxes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
    feature_maps = [[30, 40], [15, 20], [8, 10], [4, 5]]
    steps = [8, 16, 32, 64]
    priors = []

    for index in range(len(feature_maps)):
        for y in range(feature_maps[index][0]):
            for x in range(feature_maps[index][1]):
                for box_size in min_boxes[index]:
                    center_x = (x + 0.5) * steps[index] / 320
                    center_y = (y + 0.5) * steps[index] / 320
                    width = box_size / 320
                    height = box_size / 320
                    priors.append([center_x, center_y, width, height])

    priors = np.array(priors)
    boxes = np.empty_like(locations)
    boxes[:, :2] = locations[:, :2] * 0.1 * priors[:, 2:] + priors[:, :2]
    boxes[:, 2:] = np.exp(locations[:, 2:] * 0.2) * priors[:, 2:]
    boxes[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
    boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]

    scores = confidences[:, 1]
    selected = scores > conf_threshold
    box_scores = np.concatenate((boxes[selected], scores[selected, None]), axis=1)
    picked = hard_nms(box_scores, 0.5, top_k=1)

    draw = ImageDraw.Draw(image)
    for box_score in picked:
        box = box_score[:4]
        left = int(box[0] * original_width)
        top = int(box[1] * original_height)
        right = int(box[2] * original_width)
        bottom = int(box[3] * original_height)
        draw.rectangle((left, top, right, bottom), outline="lime", width=3)
        draw.text((left, top - 15), "Face: %.2f" % box_score[4], fill="lime")

    image.save(output_path)

# Edge detection
def non_maximum_suppression(magnitude, angle):
    result = np.zeros_like(magnitude)
    angle = angle * 180 / np.pi
    angle[angle < 0] += 180

    center = magnitude[1:-1, 1:-1]
    horizontal = (angle[1:-1, 1:-1] < 22.5) | (angle[1:-1, 1:-1] >= 157.5)
    diagonal = (angle[1:-1, 1:-1] >= 22.5) & (angle[1:-1, 1:-1] < 67.5)
    vertical = (angle[1:-1, 1:-1] >= 67.5) & (angle[1:-1, 1:-1] < 112.5)
    anti_diagonal = (angle[1:-1, 1:-1] >= 112.5) & (angle[1:-1, 1:-1] < 157.5)

    keep = horizontal & (center >= magnitude[1:-1, :-2]) & (center >= magnitude[1:-1, 2:])
    keep = keep | (diagonal & (center >= magnitude[2:, :-2]) & (center >= magnitude[:-2, 2:]))
    keep = keep | (vertical & (center >= magnitude[:-2, 1:-1]) & (center >= magnitude[2:, 1:-1]))
    keep = keep | (anti_diagonal & (center >= magnitude[:-2, :-2]) & (center >= magnitude[2:, 2:]))
    result[1:-1, 1:-1] = center * keep

    return result

def hysteresis_thresholding(img, low_thresh, high_thresh):
    result = np.zeros_like(img, dtype=np.uint8)
    strong = img >= high_thresh
    weak = (img >= low_thresh) & (img < high_thresh)
    result[strong] = 255
    result[weak] = 50

    for y in range(1, img.shape[0] - 1):
        for x in range(1, img.shape[1] - 1):
            if weak[y, x]:
                neighbors = strong[y - 1:y + 2, x - 1:x + 2]
                if np.any(neighbors):
                    result[y, x] = 255
                else:
                    result[y, x] = 0

    return result

def canny_scipy(image_path, output_path="edges_scipy.png", low=20, high=60):
    image = Image.open(image_path).convert("L")
    image = np.asarray(image).astype(np.float32)
    smoothed = gaussian_filter(image, sigma=1.4)

    horizontal_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    vertical_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    horizontal = convolve(smoothed, horizontal_kernel)
    vertical = convolve(smoothed, vertical_kernel)
    magnitude = np.sqrt(horizontal ** 2 + vertical ** 2)
    magnitude = magnitude / magnitude.max() * 255
    angle = np.arctan2(vertical, horizontal)

    thinned = non_maximum_suppression(magnitude, angle)
    edges = hysteresis_thresholding(thinned, low, high)
    Image.fromarray(edges).save(output_path)

if __name__ == "__main__":
    detect_faces("face.jpg")
    canny_scipy("face.jpg")