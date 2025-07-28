from PIL import Image
import pytesseract

def extract_text(image_path):
    try:
        
        img = Image.open(image_path)
          
        text = pytesseract.image_to_string(img)
        
        return text.strip() if text.strip() else None
    except Exception as e:
        print(f"Error extracting text: {e}")
        return None
