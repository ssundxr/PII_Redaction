import argparse
from PIL import Image
from .ocr import extract_text
from .detector import detect_pii
from .redactor import redact_text


def main() -> None:
    parser = argparse.ArgumentParser(prog="pii-redaction")
    parser.add_argument("image", help="Path to input image")
    args = parser.parse_args()

    img = Image.open(args.image)
    text = extract_text(img)
    detections = detect_pii(text)
    spans = [span for _, span in detections]
    redacted = redact_text(text, spans)
    print(redacted)


if __name__ == "__main__":
    main()
