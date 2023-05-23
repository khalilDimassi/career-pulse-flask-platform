# NLPackage

NLPackage is a directory that contains the necessary models and data files for running the CareerPulse NLP application. Follow the instructions below to initialize and download the contents of the NLPackage directory.

## Initializing NLPackage

To initialize NLPackage and download the required files, please follow these steps:

### Windows

1. Open the command prompt or PowerShell and navigate to the project directory (`careerpulse`).

2. Execute the following commands to create the required directories and download the files:

```shell
mkdir NLPackage\models\bert-base-uncased
mkdir NLPackage\nltk_data\corpora\stopwords
mkdir NLPackage\nltk_data\tokenizers\punkt\PY3

REM Download bert-base-uncased files
curl -LJO "https://huggingface.co/bert-base-uncased\config.json"
curl -LJO "https://huggingface.co/bert-base-uncased\pytorch_model.bin"
curl -LJO "https://huggingface.co/bert-base-uncased\special_tokens_map.json"
curl -LJO "https://huggingface.co/bert-base-uncased\tokenizer_config.json"
curl -LJO "https://huggingface.co/bert-base-uncased\vocab.txt"

REM Download nltk_data files
curl -LJO "URL_TO_stopwords.zip"
unzip stopwords.zip -d NLPackage\nltk_data\corpora\stopwords
curl -LJO "URL_TO_punkt.zip"
unzip punkt.zip -d NLPackage\nltk_data\tokenizers\punkt\PY3

```

3. Update the file paths in app.py to point to the downloaded files.

## Prepare the Envirement

1. Make sure you have Python installed on your system. You can check if you have Python installed by running the following command in your terminal:

```shell 
python --version
```

2. Create a virtual environment for the project and Activate the virtual environment using the following command:


```shell 
python -m venv venv

./venv/bin/Activate.ps1

```

3. To install the packages listed in the requirements.txt file, use the following command:

```shell 
pip install -r requirements.txt

```

This will install all the necessary packages for the application.
(you might need to install tesseract ocr and add it to system path, add path to app.py)

## Running the Application

After initializing NLPackage, you can run the CareerPulse application. Make sure you have the required dependencies installed and set up a MySQL database using the provided `SQLALCHEMY_DATABASE_URI`.

Execute the following command to start the application:

```shell
python -m app
```

Open your web browser and navigate to `http://localhost:5000` to access the CareerPulse application.

