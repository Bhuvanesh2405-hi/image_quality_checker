import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  THRESHOLDS  — tuned for mobile-captured / scanned Aadhaar & PAN cards
#  Adjust MIN_SHARPNESS_* if you see too many false-fails on your corpus.
# ══════════════════════════════════════════════════════════════════════════════

# Resolution
MIN_DIM_HARD            = 200       # px  — PaddleOCR det needs at least this

# Sharpness  (three independent metrics — ALL must pass)
MIN_LAPLACIAN_VAR       = 120.0     # raw Laplacian variance (was 80 normalised — too lenient)
MIN_TENENGRAD           = 80.0      # Sobel energy — catches motion blur Laplacian misses
MIN_FFT_SHARPNESS       = 10.0      # high-frequency energy ratio from FFT

# Contrast
MIN_CONTRAST_HARD       = 20.0      # std-dev — moved from warning → hard fail
MIN_CONTRAST_WARN       = 35.0      # soft warning before hard-fail zone

# Brightness
BRIGHTNESS_DARK_HARD    = 30.0      # pitch-black
BRIGHTNESS_BRIGHT_HARD  = 252.0     # completely blown-out (was warning only)
BRIGHTNESS_BRIGHT_WARN  = 230.0     # over-exposed warning

# Noise
NOISE_HARD              = 35.0      # heavy artefacts that corrupt OCR
NOISE_WARN              = 20.0      # moderate noise


# ══════════════════════════════════════════════════════════════════════════════
#  SHARPNESS METRICS
# ══════════════════════════════════════════════════════════════════════════════

def _laplacian_variance(gray: np.ndarray) -> float:
    """Classic blur detector. Fast, reliable for out-of-focus blur."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _tenengrad(gray: np.ndarray) -> float:
    """
    Sobel-based focus measure (Tenengrad).
    Better than Laplacian for motion blur & partial focus.
    Returns mean gradient energy (normalised by pixel count).
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_energy = gx ** 2 + gy ** 2
    return float(np.mean(gradient_energy))


def _fft_sharpness(gray: np.ndarray) -> float:
    """
    High-frequency energy ratio via FFT.
    Blurry images have most energy in low frequencies.
    Returns % of energy in the outer 30 % of the frequency spectrum.
    """
    h, w  = gray.shape
    fft   = np.fft.fft2(gray.astype(np.float32))
    fft_s = np.fft.fftshift(fft)
    mag   = np.abs(fft_s)

    # Create a mask for the HIGH-frequency ring (outer 30 %)
    cy, cx = h // 2, w // 2
    r_max  = min(cy, cx)
    r_low  = int(r_max * 0.70)          # inner radius  — low-freq centre

    Y, X   = np.ogrid[:h, :w]
    dist   = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    high_freq_mask = dist >= r_low
    total_energy   = np.sum(mag) + 1e-6
    high_freq_ratio = np.sum(mag[high_freq_mask]) / total_energy

    return float(high_freq_ratio * 100)   # percentage


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def check_clarity(image_bytes: bytes) -> dict:
    """
    Multi-metric image quality check tuned for PaddleOCR on KYC documents.

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
    img        = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        logger.error("[CLARITY] Failed to decode image bytes")
        return {
            "verdict"    : "not_clear",
            "hard_fails" : ["decode_failed"],
            "warnings"   : [],
            "metrics"    : {}
        }

    h, w    = img.shape[:2]
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    min_dim = min(h, w)

    # ── Compute metrics ───────────────────────────────────────────────────────

    lap_var     = _laplacian_variance(gray)
    tenengrad   = _tenengrad(gray)
    fft_sharp   = _fft_sharpness(gray)
    contrast    = float(gray.std())
    brightness  = float(gray.mean())

    blurred     = cv2.GaussianBlur(gray, (5, 5), 0)
    noise       = float(np.std(
        gray.astype(np.float32) - blurred.astype(np.float32)
    ))

    metrics = {
        "laplacian_variance" : round(lap_var,    2),
        "tenengrad"          : round(tenengrad,  2),
        "fft_sharpness_pct"  : round(fft_sharp,  2),
        "contrast"           : round(contrast,   2),
        "brightness"         : round(brightness, 2),
        "noise"              : round(noise,       2),
        "resolution"         : f"{w}x{h}",
    }

    # ── Hard-fail evaluation ──────────────────────────────────────────────────
    hard_fails = []
    warnings   = []

    # --- Resolution ---
    if min_dim < MIN_DIM_HARD:
        hard_fails.append(
            f"too_small (min_dim={min_dim}px, need >={MIN_DIM_HARD}px)"
        )

    # --- Sharpness (all three must pass) ---
    if lap_var < MIN_LAPLACIAN_VAR:
        hard_fails.append(
            f"blurry_laplacian (lap_var={lap_var:.1f}, need >={MIN_LAPLACIAN_VAR})"
        )

    if tenengrad < MIN_TENENGRAD:
        hard_fails.append(
            f"blurry_tenengrad (tenengrad={tenengrad:.1f}, need >={MIN_TENENGRAD})"
        )

    if fft_sharp < MIN_FFT_SHARPNESS:
        hard_fails.append(
            f"blurry_fft (fft_sharpness={fft_sharp:.1f}%, need >={MIN_FFT_SHARPNESS}%)"
        )

    # --- Contrast (now a hard fail, not just a warning) ---
    if contrast < MIN_CONTRAST_HARD:
        hard_fails.append(
            f"low_contrast (contrast={contrast:.1f}, need >={MIN_CONTRAST_HARD})"
        )
    elif contrast < MIN_CONTRAST_WARN:
        warnings.append(
            f"low_contrast_warn (contrast={contrast:.1f}, expected >={MIN_CONTRAST_WARN})"
        )

    # --- Brightness ---
    if brightness < BRIGHTNESS_DARK_HARD:
        hard_fails.append(
            f"too_dark (brightness={brightness:.1f}, need >={BRIGHTNESS_DARK_HARD})"
        )

    if brightness > BRIGHTNESS_BRIGHT_HARD:
        hard_fails.append(
            f"overexposed (brightness={brightness:.1f}, threshold={BRIGHTNESS_BRIGHT_HARD})"
        )
    elif brightness > BRIGHTNESS_BRIGHT_WARN:
        warnings.append(
            f"very_bright (brightness={brightness:.1f}, threshold={BRIGHTNESS_BRIGHT_WARN})"
        )

    # --- Noise ---
    if noise > NOISE_HARD:
        hard_fails.append(
            f"too_noisy (noise={noise:.1f}, threshold={NOISE_HARD})"
        )
    elif noise > NOISE_WARN:
        warnings.append(
            f"noisy (noise={noise:.1f}, threshold={NOISE_WARN})"
        )

    # ── Verdict ───────────────────────────────────────────────────────────────
    verdict = "not_clear" if hard_fails else "clear"

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


# ── Convenience wrapper ───────────────────────────────────────────────────────

def is_image_clear(image_bytes: bytes) -> bool:
    """Boolean wrapper. Returns True if safe to send to PaddleOCR."""
    return check_clarity(image_bytes)["verdict"] == "clear"
