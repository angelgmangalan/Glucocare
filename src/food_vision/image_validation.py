"""In-memory validation and privacy-preserving food image normalization."""

from __future__ import annotations

import io
import os
import warnings
from dataclasses import dataclass
from typing import BinaryIO

from PIL import Image, ImageOps, UnidentifiedImageError


SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_DIMENSION = 2048
MAX_IMAGE_PIXELS = 40_000_000


def configured_max_upload_bytes() -> int:
    raw = os.getenv("FOOD_IMAGE_MAX_UPLOAD_MB", "10")
    try:
        megabytes = int(raw)
    except ValueError as error:
        raise ValueError("FOOD_IMAGE_MAX_UPLOAD_MB must be a positive integer.") from error
    if megabytes < 1 or megabytes > 50:
        raise ValueError("FOOD_IMAGE_MAX_UPLOAD_MB must be between 1 and 50.")
    return megabytes * 1024 * 1024


@dataclass(frozen=True)
class PreparedImage:
    data: bytes
    mime_type: str
    width: int
    height: int


def _read_upload(upload: bytes | BinaryIO, max_bytes: int) -> bytes:
    if isinstance(upload, bytes):
        content = upload
    else:
        try:
            upload.seek(0)
            content = upload.read(max_bytes + 1)
        except (AttributeError, OSError) as error:
            raise ValueError("The uploaded file could not be read. Please try again.") from error
    if not content:
        raise ValueError("The uploaded image is empty.")
    if len(content) > max_bytes:
        raise ValueError(f"The image exceeds the {max_bytes // (1024 * 1024)} MB upload limit.")
    return content


def prepare_image(
    upload: bytes | BinaryIO,
    filename: str,
    *,
    max_bytes: int = MAX_UPLOAD_BYTES,
    max_dimension: int = MAX_IMAGE_DIMENSION,
) -> PreparedImage:
    """Validate extension and decoded format, strip metadata, orient, and resize."""
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    expected_format = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "webp": "WEBP"}.get(extension)
    if expected_format is None:
        raise ValueError("Unsupported image format. Upload a JPEG, PNG, or WEBP image.")
    if max_bytes < 1 or max_dimension < 1:
        raise ValueError("Image size settings must be positive.")
    content = _read_upload(upload, max_bytes)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as source:
                actual_format = source.format
                if actual_format not in SUPPORTED_FORMATS or actual_format != expected_format:
                    raise ValueError("The image content does not match its file extension.")
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise ValueError("The image dimensions exceed the safe processing limit.")
                source.load()
                oriented = ImageOps.exif_transpose(source)
                if oriented.width < 1 or oriented.height < 1:
                    raise ValueError("The uploaded image has invalid dimensions.")
                if oriented.mode in {"RGBA", "LA"} or "transparency" in oriented.info:
                    rgba = oriented.convert("RGBA")
                    background = Image.new("RGB", rgba.size, "white")
                    background.paste(rgba, mask=rgba.getchannel("A"))
                    normalized = background
                else:
                    normalized = oriented.convert("RGB")
                normalized.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                output = io.BytesIO()
                normalized.save(output, format="JPEG", quality=90, optimize=True)
                return PreparedImage(
                    data=output.getvalue(),
                    mime_type="image/jpeg",
                    width=normalized.width,
                    height=normalized.height,
                )
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise ValueError("The uploaded file is not a valid, safely decodable image.") from error
