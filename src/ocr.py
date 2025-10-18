from typing import List, Tuple
from PIL import Image
import pytesseract


def extract_text(image: Image.Image) -> str:
    return pytesseract.image_to_string(image)


def extract_words_with_boxes(image: Image.Image) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words: List[Tuple[str, Tuple[int, int, int, int]]] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = data["text"][i]
        if text:
            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])
            words.append((text, (x, y, w, h)))
    return words
