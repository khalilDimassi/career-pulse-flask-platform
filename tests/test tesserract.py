import pytesseract
from PIL import Image

# Load an example image
image_path = r'uploads\offers\offer-test.png'
image = Image.open(image_path)

# Use Tesseract OCR to extract text from the image
text = pytesseract.image_to_string(image)

# Print the extracted text
print(text)
