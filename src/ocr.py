from typing import List, Tuple
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import logging
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import os
import warnings

# Suppress TrOCR warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", message=".*early_stopping.*")


# Configure Tesseract path for Windows (uncomment if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\sdshy\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'


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


class TrOCRExtractor:
    """
    A text extraction class that combines pytesseract for text line detection
    with Microsoft's TrOCR model for accurate text recognition.
    
    This class uses pytesseract to detect text line bounding boxes and then
    applies TrOCR (Transformer-based OCR) to each detected line for improved
    text recognition accuracy.
    """
    
    def __init__(self, model_name: str = "microsoft/trocr-base-handwritten"):
        """
        Initialize the TrOCRExtractor with the specified TrOCR model.
        
        Args:
            model_name (str): The name of the TrOCR model to use.
                            Defaults to "microsoft/trocr-base-handwritten".
        
        Raises:
            Exception: If the model fails to load.
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        
        try:
            self.logger.info(f"Loading TrOCR model: {model_name}")
            self.processor = TrOCRProcessor.from_pretrained(model_name, use_fast=True)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            
            # Set device (GPU if available, otherwise CPU)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.logger.info(f"TrOCR model loaded successfully on device: {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load TrOCR model {model_name}: {str(e)}")
            raise Exception(f"Failed to initialize TrOCRExtractor: {str(e)}")
    
    def extract_text_lines(self, image: Image.Image, beams: int = 1) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        """
        Extract text from an image by detecting text lines and applying TrOCR to each line.
        
        This method uses pytesseract to detect text line bounding boxes, crops each line
        from the original image, and then applies TrOCR for accurate text recognition.
        
        Args:
            image (PIL.Image.Image): The input image to extract text from.
            beams (int): Number of beams for TrOCR generation (default 1 for speed).
        
        Returns:
            List[Tuple[str, Tuple[int, int, int, int]]]: A list of tuples containing
            the extracted text and its bounding box coordinates (x, y, width, height).
        
        Raises:
            Exception: If text extraction fails.
        """
        try:
            self.logger.info("Starting text line extraction")
            
            # Use pytesseract to detect text line bounding boxes
            # psm 6 assumes a single uniform block of text
            data = pytesseract.image_to_data(
                image, 
                output_type=pytesseract.Output.DICT,
                config='--psm 6'
            )
            
            # Group words into lines based on their y-coordinates
            lines = self._group_words_into_lines(data)
            
            results = []
            
            if not lines:
                return results
            
            # Prepare line images for batch processing
            line_images = []
            valid_boxes = []
            
            for line_bbox in lines:
                try:
                    # Crop the line from the original image
                    x, y, w, h = line_bbox
                    line_image = image.crop((x, y, x + w, y + h))
                    
                    # Preprocess the line image: grayscale, contrast, smooth, and conditionally scale
                    line_image = line_image.convert('L')
                    enhancer = ImageEnhance.Contrast(line_image)
                    line_image = enhancer.enhance(1.5)
                    line_image = line_image.filter(ImageFilter.SMOOTH)
                    if line_image.width < 100 or line_image.height < 100:
                        # Scale by 2x
                        new_size = (line_image.width * 2, line_image.height * 2)
                        line_image = line_image.resize(new_size, Image.BICUBIC)
                    line_image = line_image.convert('RGB')
                    
                    line_images.append(line_image)
                    valid_boxes.append(line_bbox)
                
                except Exception as e:
                    self.logger.warning(f"Failed to process line at {line_bbox}: {str(e)}")
                    continue
            
            # Process all lines in batches
            batch_size = 8
            for i in range(0, len(line_images), batch_size):
                batch = line_images[i:i+batch_size]
                batch_texts = self._extract_text_with_trocr(batch, beams)
                
                for text, bbox in zip(batch_texts, valid_boxes[i:i+batch_size]):
                    if text.strip():  # Only add non-empty text
                        results.append((text.strip(), bbox))
                        self.logger.debug(f"Extracted text: '{text.strip()}' at bbox: {bbox}")
            
            self.logger.info(f"Successfully extracted {len(results)} text lines")
            return results
            
        except Exception as e:
            self.logger.error(f"Text line extraction failed: {str(e)}")
            raise Exception(f"Failed to extract text lines: {str(e)}")
    
    def _group_words_into_lines(self, data: dict) -> List[Tuple[int, int, int, int]]:
        """
        Group detected words into text lines based on their y-coordinates.
        
        Args:
            data (dict): Pytesseract output data containing word positions.
        
        Returns:
            List[Tuple[int, int, int, int]]: List of line bounding boxes (x, y, w, h).
        """
        words = []
        n = len(data.get("text", []))
        
        for i in range(n):
            text = data["text"][i]
            if text.strip():  # Only consider non-empty text
                x = int(data["left"][i])
                y = int(data["top"][i])
                w = int(data["width"][i])
                h = int(data["height"][i])
                words.append((text, x, y, w, h))
        
        if not words:
            return []
        
        # Sort words by y-coordinate (top to bottom)
        words.sort(key=lambda word: word[2])
        
        lines = []
        current_line_words = [words[0]]
        
        # Group words into lines based on y-coordinate proximity
        line_height_threshold = 10  # pixels
        
        for word in words[1:]:
            current_y = word[2]
            last_y = current_line_words[-1][2]
            
            if abs(current_y - last_y) <= line_height_threshold:
                # Same line
                current_line_words.append(word)
            else:
                # New line
                if current_line_words:
                    lines.append(self._calculate_line_bbox(current_line_words))
                current_line_words = [word]
        
        # Add the last line
        if current_line_words:
            lines.append(self._calculate_line_bbox(current_line_words))
        
        return lines
    
    def _calculate_line_bbox(self, words: List[Tuple[str, int, int, int, int]]) -> Tuple[int, int, int, int]:
        """
        Calculate the bounding box for a line of words.
        
        Args:
            words (List[Tuple[str, int, int, int, int]]): List of words with their positions.
        
        Returns:
            Tuple[int, int, int, int]: Line bounding box (x, y, w, h).
        """
        if not words:
            return (0, 0, 0, 0)
        
        # Find the leftmost x, topmost y, rightmost x+w, and bottommost y+h
        min_x = min(word[1] for word in words)
        min_y = min(word[2] for word in words)
        max_x = max(word[1] + word[3] for word in words)
        max_y = max(word[2] + word[4] for word in words)
        
        return (min_x, min_y, max_x - min_x, max_y - min_y)
    
    def _extract_text_with_trocr(self, images: List[Image.Image], beams: int = 1) -> List[str]:
        """
        Extract text from a batch of images using TrOCR.
        
        Args:
            images: List of PIL Image objects to process
            beams: Number of beams for generation (1 for speed, 2+ for accuracy)
        
        Returns:
            List of extracted text strings
        """
        try:
            # Preprocess all images
            pixel_values = self.processor(images, return_tensors="pt").pixel_values.to(self.device)
            
            # Generate text with batch processing
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    num_beams=beams,
                    max_new_tokens=64,
                    do_sample=False,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode all generated texts
            generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            return generated_texts
            
        except Exception as e:
            self.logger.error(f"Batch text extraction failed: {str(e)}")
            return [""] * len(images)


# Test function (optional)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        extractor = TrOCRExtractor()
        print("TrOCRExtractor initialized successfully")
    except Exception as e:
        print(f"Error: {e}")
