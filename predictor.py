import numpy as np
import cv2
from io import BytesIO
def check_clarity(image_bytes):
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return "not_clear"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    THRESHOLD = 120.0
    if laplacian_var > THRESHOLD:
        return "clear"
    else:
        return "not_clear"