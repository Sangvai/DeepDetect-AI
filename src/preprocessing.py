"""Shared preprocessing used by both dataset preparation and inference.

Using the exact same functions in training-data prep and in predict.py is what
makes the model's predictions valid at inference time -- if the crop, resize,
or normalization differed even slightly, the model would see inputs unlike
anything it trained on.
"""
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

IMG_SIZE = 224
SAVE_SIZE = 256  # stored slightly larger than IMG_SIZE so training can RandomResizedCrop for augmentation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_face_box(image_bgr: np.ndarray):
    """Returns (x, y, w, h) of the largest detected face, or None if no face found."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    # largest face by area, in case of multiple detections
    return max(faces, key=lambda f: f[2] * f[3])


def crop_face(image_bgr: np.ndarray, box, margin: float = 0.3) -> np.ndarray:
    """Crops the face with extra margin so hair/jaw aren't clipped, then resizes to SAVE_SIZE."""
    x, y, w, h = box
    mx, my = int(w * margin), int(h * margin)
    H, W = image_bgr.shape[:2]
    x0, y0 = max(x - mx, 0), max(y - my, 0)
    x1, y1 = min(x + w + mx, W), min(y + h + my, H)
    face = image_bgr[y0:y1, x0:x1]
    return cv2.resize(face, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_AREA)


def detect_and_crop_face(image_bgr: np.ndarray):
    """Full detect+crop step. Returns a SAVE_SIZE x SAVE_SIZE BGR face image, or None if no face found."""
    box = detect_face_box(image_bgr)
    if box is None:
        return None
    return crop_face(image_bgr, box)


def get_train_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    """Augmentation only makes sense at training time -- it teaches the model to
    ignore small pose/lighting/framing changes instead of memorizing exact pixels."""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_eval_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    """No randomness here -- validation/test/inference must be deterministic and
    must match what the pretrained backbone expects (ImageNet mean/std)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def preprocess_for_inference(image_bgr: np.ndarray):
    """Runs face detection + crop + eval transform on a raw uploaded image.

    Returns (face_rgb_uint8, input_tensor) where face_rgb_uint8 is for display
    and input_tensor is a (1, 3, IMG_SIZE, IMG_SIZE) batch ready for the model.
    Returns (None, None) if no face was detected.
    """
    face_bgr = detect_and_crop_face(image_bgr)
    if face_bgr is None:
        return None, None
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb)
    tensor = get_eval_transforms()(pil_img).unsqueeze(0)
    return face_rgb, tensor
