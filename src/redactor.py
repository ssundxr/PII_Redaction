from typing import List, Tuple
from PIL import Image, ImageDraw


def redact_image(image: Image.Image, boxes: List[Tuple[int, int, int, int]], fill=(0, 0, 0)) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for x, y, w, h in boxes:
        draw.rectangle([x, y, x + w, y + h], fill=fill)
    return img


def redact_text(text: str, spans: List[Tuple[int, int]], mask: str = "[*REDACTED*]") -> str:
    if not spans:
        return text
    spans = sorted(spans, key=lambda s: s[0])
    out: List[str] = []
    i = 0
    for start, end in spans:
        out.append(text[i:start])
        out.append(mask)
        i = end
    out.append(text[i:])
    return "".join(out)
