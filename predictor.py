import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

def check_clarity(image_bytes: bytes) -> dict:
    """
    Multi-metric image quality check tuned for PaddleOCR on KYC documents
    (Aadhaar / PAN cards).

    Returns:
        {
            "verdict"    : "clear" | "not_clear",
            "hard_fails" : [...],   # blocked from OCR
            "warnings"   : [...],   # logged but OCR still attempted
            "metrics"    : {...}    # raw numbers for debugging
        }
    """

    # ── Decode ────────────────────────────────────────────────────────────────
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        logger.error("[CLARITY] Failed to decode image bytes")
        return {
            "verdict"    : "not_clear",
            "hard_fails" : ["decode_failed"],
            "warnings"   : [],
            "metrics"    : {}
        }

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    min_dim = min(h, w)

    # ── Metrics ───────────────────────────────────────────────────────────────

    # 1. Sharpness — Laplacian variance normalised by resolution
    #    Normalising prevents a 4K scan and a 640px crop from needing
    #    different thresholds.
    laplacian_var         = cv2.Laplacian(gray, cv2.CV_64F).var()
    normalised_sharpness  = laplacian_var / max(min_dim / 1000.0, 0.1)

    # 2. Contrast — std-dev of pixel intensities
    #    Low std-dev → washed-out or photocopied doc
    contrast = float(gray.std())

    # 3. Brightness — mean pixel value
    brightness = float(gray.mean())

    # 4. Noise — diff between raw and mildly blurred image
    #    Heavy sensor noise confuses PaddleOCR's det model
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise   = float(np.std(gray.astype(np.float32) - blurred.astype(np.float32)))

    # ── Thresholds ────────────────────────────────────────────────────────────
    #
    #   Hard-fail  → image blocked; do not send to OCR
    #   Warning    → suspicious; OCR still attempted; logged for monitoring
    #
    #   Values tuned for mobile-captured / scanned Aadhaar & PAN documents.
    #   Re-evaluate against your own sample set if doc types change.

    MIN_DIM_HARD          = 150    # px  — below this PaddleOCR det cannot box text
    SHARPNESS_HARD        = 80.0   # normalised units
    BRIGHTNESS_DARK_HARD  = 40.0   # mean px value — pitch-black image

    CONTRAST_WARN         = 25.0   # below → faded/low-ink doc (OCR still worth trying)
    BRIGHTNESS_BRIGHT_WARN= 245.0  # above → over-exposed (white bg docs hit ~235, so soft)
    NOISE_WARN            = 25.0   # above → scanner grain or heavy JPEG artefacts

    # ── Evaluation ────────────────────────────────────────────────────────────
    hard_fails = []
    warnings   = []

    if min_dim < MIN_DIM_HARD:
        hard_fails.append(f"too_small (min_dim={min_dim}px, need >={MIN_DIM_HARD}px)")

    if normalised_sharpness < SHARPNESS_HARD:
        hard_fails.append(
            f"blurry (sharpness={normalised_sharpness:.1f}, need >={SHARPNESS_HARD})"
        )

    if brightness < BRIGHTNESS_DARK_HARD:
        hard_fails.append(
            f"too_dark (brightness={brightness:.1f}, need >={BRIGHTNESS_DARK_HARD})"
        )

    if contrast < CONTRAST_WARN:
        warnings.append(
            f"low_contrast (contrast={contrast:.1f}, expected >={CONTRAST_WARN})"
        )

    if brightness > BRIGHTNESS_BRIGHT_WARN:
        warnings.append(
            f"very_bright (brightness={brightness:.1f}, threshold={BRIGHTNESS_BRIGHT_WARN})"
        )

    if noise > NOISE_WARN:
        warnings.append(
            f"noisy (noise={noise:.1f}, threshold={NOISE_WARN})"
        )

    # ── Verdict ───────────────────────────────────────────────────────────────
    verdict = "not_clear" if hard_fails else "clear"

    metrics = {
        "sharpness_raw"        : round(laplacian_var, 2),
        "sharpness_normalised" : round(normalised_sharpness, 2),
        "contrast"             : round(contrast, 2),
        "brightness"           : round(brightness, 2),
        "noise"                : round(noise, 2),
        "resolution"           : f"{w}x{h}",
    }

    # ── Logging ───────────────────────────────────────────────────────────────
    if hard_fails:
        logger.warning(
            "[CLARITY] BLOCKED | hard_fails=%s | warnings=%s | metrics=%s",
            hard_fails, warnings, metrics
        )
    elif warnings:
        logger.info(
            "[CLARITY] PASS WITH WARNINGS | warnings=%s | metrics=%s",
            warnings, metrics
        )
    else:
        logger.debug("[CLARITY] PASS | metrics=%s", metrics)

    return {
        "verdict"    : verdict,
        "hard_fails" : hard_fails,
        "warnings"   : warnings,
        "metrics"    : metrics,
    }


# ── Convenience wrapper (drop-in replacement for your original) ───────────────

def is_image_clear(image_bytes: bytes) -> bool:
    """
    Simple boolean wrapper around check_clarity().
    Returns True if the image is safe to send to PaddleOCR.
    """
    return check_clarity(image_bytes)["verdict"] == "clear"
