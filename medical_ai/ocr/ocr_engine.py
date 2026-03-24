import cv2
import pytesseract
import numpy as np
from pdf2image import convert_from_path
import os

class OCREngine:
    def __init__(self, tesseract_cmd=None):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def preprocess_image(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Apply thresholding to preprocess the image
        # Using adaptive thresholding for varying illumination
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Denoising
        denoised = cv2.fastNlMeansDenoising(thresh, None, 30, 7, 21)
        return denoised

    def extract_text_from_image(self, image_path):
        """Extracts text from a single image."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            processed_image = self.preprocess_image(image)
            text = pytesseract.image_to_string(processed_image)
            return text
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return ""

    def extract_text_from_pdf(self, pdf_path):
        """Extracts text from a PDF document by converting pages to images."""
        try:
            images = convert_from_path(pdf_path)
            full_text = ""
            for idx, img in enumerate(images):
                # Convert PIL image to OpenCV format
                open_cv_image = np.array(img)
                # Convert RGB to BGR 
                open_cv_image = open_cv_image[:, :, ::-1].copy()
                
                processed_image = self.preprocess_image(open_cv_image)
                text = pytesseract.image_to_string(processed_image)
                full_text += f"\n--- Page {idx + 1} ---\n{text}"
                
            return full_text
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            return ""

    def extract_text(self, file_path):
        """Identifies file type and extracts text."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            return self.extract_text_from_image(file_path)
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported file format: {ext}")

if __name__ == "__main__":
    # Example usage:
    print("OCR Engine Initialized.")
