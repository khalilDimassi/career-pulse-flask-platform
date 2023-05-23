import torch
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from pdf2image import convert_from_path
import imghdr


def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    binary_image = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    coords = np.column_stack(np.where(binary_image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (height, width) = gray_image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(binary_image, rotation_matrix, (
        width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return rotated_image

def extract_text(image, tesseract_config):
    text = pytesseract.image_to_string(image, config=tesseract_config, lang='eng')
    return text


def extract_text_offers(file_path, tesseract_config):
    # Check if the file is an image
    if imghdr.what(file_path):
        image = cv2.imread(file_path)
        preprocessed_image = preprocess_image(image)
        return extract_text(preprocessed_image, tesseract_config)

    # Check if the file is a PDF
    if file_path.lower().endswith('.pdf'):
        # Convert PDF pages to images
        images = convert_from_path(file_path)

        # Extract text from each image
        extracted_text = []
        for image in images:
            preprocessed_image = preprocess_image(np.array(image))
            text = extract_text(preprocessed_image, tesseract_config)
            extracted_text.append(text)

        # Concatenate all extracted text
        return '\n'.join(extracted_text)

    # Return empty string if the file is neither an image nor a PDF
    return ''

# Set NLTK data path to your custom directory
nltk.data.path.append("NLPackage/models/nltk_data")

def clean_text(text):
    # remove special characters and digits
    text = re.sub(r'\W+', ' ', text.lower())
    # tokenize the text
    tokens = nltk.word_tokenize(text)
    # remove stop words and perform lemmatization
    stop_words = nltk.corpus.stopwords.words('english')
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    # rejoin the tokens into a string
    clean_text = ' '.join(tokens)
    return clean_text

# Define a function to generate contextual embeds from raw text


def encode_text(text, model, tokenizer, adaptive_pool):
    input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True,
                             max_length=512, padding='max_length', truncation=True)).unsqueeze(0)
    outputs = model(input_ids)
    last_hidden_state = outputs[0].squeeze(0)

    # Apply Adaptive Average Pooling
    last_hidden_state_pooled = adaptive_pool(
        last_hidden_state.permute(1, 0)).permute(1, 0)

    mean_pooling = torch.mean(last_hidden_state_pooled, dim=0)
    return mean_pooling.detach().numpy()

