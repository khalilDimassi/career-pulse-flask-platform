from transformers import pipeline, BertTokenizer, BertModel
import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration

def test_models():
    # Test tesseract_config
    tesseract_config = r'--oem 3 --psm 6 --tessdata-dir ./NLPackage/models'
    print("success!")

# Call the test function
test_models()
