# Database related imports
import mysql.connector
from flask_sqlalchemy import SQLAlchemy

# File handling related imports
import os
from werkzeug.utils import secure_filename

# Web framework related imports
from flask import Flask, render_template, request

# NLP related imports
import spacy
import torch.nn as nn
import pytesseract
from spacy.matcher import PhraseMatcher
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor
# , BartTokenizer, BartForConditionalGeneration
from transformers import pipeline, BertTokenizer, BertModel
from NLPackage.methods import extract_text_offers, clean_text, encode_text


app = Flask(__name__)

# TODO: apply your own mySQL connector credentioals  
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:2321999@localhost/careerpulsedb'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = r'uploads'


# Initialize SQLAlchemy
db = SQLAlchemy(app)

# initialize tesserract model: install pytesserract from the official git repos and add it to path: RESRTART IS NAICESSARRY

# Load Spacy model
nlp = spacy.load("en_core_web_lg")

# Initialize Skill Extractor
skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)

# Initialize BERT model & BERT Tokenizer and limit the pooling

tokenizer = BertTokenizer.from_pretrained('NLPackage/models/bert-base-uncased')
model = BertModel.from_pretrained('NLPackage/models/bert-base-uncased')
adaptive_pool = nn.AdaptiveAvgPool1d(1024)


def file_exists(file_path):
    return os.path.exists(file_path)


# Offer model
class Offer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filepath = db.Column(db.String(255))
    text = db.Column(db.Text)
    embeds = db.Column(db.Text)
    geo = db.Column(db.String(255))
    offer_type = db.Column(db.String(255))
    time = db.Column(db.String(255))
    active = db.Column(db.Boolean)
    poster_company = db.Column(db.String(255))
    skills = db.Column(db.String(255))
    industry = db.Column(db.String(255))

# Resume model
class Resume(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filepath = db.Column(db.String(255))
    text = db.Column(db.Text)
    embeds = db.Column(db.Text)
    geo = db.Column(db.String(255))
    profession = db.Column(db.String(255))
    industry = db.Column(db.String(255))
    skills = db.Column(db.String(255))

# Home page
@app.route('/')
def home():
    return render_template('home.html')

# Post offer
@app.route('/post_offer', methods=['GET', 'POST'])
def post_offer():
    if request.method == 'POST':
        # Get form data
        text = request.form['text']
        geo = request.form['geo']
        offer_type_v = request.form['type']
        time = request.form['time']
        active = request.form.get('active') == 'on'  # Convert to boolean
        poster_company = request.form['poster_company']
        industry = request.form['industry']

        # Check if a file was uploaded
        if 'file' not in request.files:
            return 'No file uploaded'

        file = request.files['file']

        # Save the uploaded file
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(
                app.config['UPLOAD_FOLDER'], 'offers', filename)
            file.save(file_path)

            # Extract text from the file
            if not text and file_exists(file_path):
                # Replace with your custom method to extract text from the file
                text = extract_text_offers(file_path)
            else:
                text += '\n' + extract_text_offers(file_path)

        # Clean the text
        clean_txt = clean_text(text)

        # Create BERT tokens and embeddings
        embeddings = encode_text(clean_txt, model, tokenizer, adaptive_pool)

        # Extract skills from the cleaned text
        doc = nlp(clean_txt)
        annotations = skill_extractor.annotate(doc.text)
        skills = [skill['doc_node_value'] for skill in annotations['results']['full_matches']] + [skill['doc_node_value'] for skill in annotations['results']['ngram_scored']]

        embeddings_string = ','.join(map(str, embeddings))
        skills_string = ','.join(map(str, skills))    

        # Save offer to the database
        offer = Offer(
            filepath=file_path if file else '',
            text=text,
            embeds=embeddings_string,
            geo=geo,
            offer_type=offer_type_v,
            time=time,
            active=active,
            poster_company=poster_company,
            skills=skills_string,
            industry=industry
        )

        db.session.add(offer)
        db.session.commit()

        return 'Offer posted successfully'

    return render_template('post_offer.html')

# Post resume
@app.route('/post_resume', methods=['GET', 'POST'])
def post_resume():
    if request.method == 'POST':
        # Get form data
        geo = request.form['geo']
        profession = request.form['profession']
        industry = request.form['industry']

        # Check if a file was uploaded
        if 'file' not in request.files:
            return 'No file uploaded'

        file = request.files['file']

        # Save the uploaded file
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(
                app.config['UPLOAD_FOLDER'], 'resumes', filename)
            file.save(file_path)

            # Extract text from the file
            text = extract_text_offers(file_path)

        # Clean the text
        clean_txt = clean_text(text)

        # Create BERT tokens and embeddings
        embeddings = encode_text(clean_txt, model, tokenizer, adaptive_pool)

        # Extract skills from the cleaned text
        doc = nlp(clean_txt)
        annotations = skill_extractor.annotate(doc.text)
        skills = [skill['doc_node_value'] for skill in annotations['results']['full_matches']] + [skill['doc_node_value'] for skill in annotations['results']['ngram_scored']]

        embeddings_string = ','.join(map(str, embeddings))
        skills_string = ','.join(map(str, skills))    

        # Save resume to the database
        resume = Resume(
            filepath=file_path if file else '',
            text=text,
            embeds=embeddings_string,
            geo=geo,
            profession=profession,
            industry=industry,
            skills=skills_string
        )
        db.session.add(resume)
        db.session.commit()

        return 'Resume posted successfully'

    return render_template('post_resume.html')

# All offers
@app.route('/all_offers')
def all_offers():
    offers = Offer.query.all()
    return render_template('all_offers.html', offers=offers)

# All resumes
@app.route('/all_resumes')
def all_resumes():
    resumes = Resume.query.all()
    return render_template('all_resumes.html', resumes=resumes)

# Dashboard
@app.route('/dashboard')
def dashboard():
    # Add code for dashboard view
    offers = Offer.query.all()
    resumes = Resume.query.all()

    return render_template('dashboard.html', resumes=resumes, offers=offers)


if __name__ == '__main__':
    # Create database tables
    with app.app_context():
        db.create_all()

    # Run the Flask application
    app.run(debug=True)
