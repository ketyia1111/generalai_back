import os
import datetime
import random
import string
import json
import requests
import psycopg2
from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for, jsonify, escape,abort)
from flask_restful import Resource, Api
from flask_cors import CORS
from hashids import Hashids
from openai import AzureOpenAI
from flask_sqlalchemy import SQLAlchemy
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv


app = Flask(__name__)
api = Api(app)
CORS(app)

load_dotenv()

app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI')
db = SQLAlchemy(app)

client = AzureOpenAI(
  azure_endpoint = os.getenv('AZURE_ENDPOINT'), 
  api_key=os.getenv('API_KEY'),  
  api_version=os.getenv('API_VERSION')
)

images = AzureOpenAI(
    api_version=os.getenv('API_VERSION'),
    azure_endpoint=os.getenv('AZURE_ENDPOINT') + "openai/deployments/Dalle3/images/generations?api-version=2023-06-01-preview",
    api_key=os.getenv('API_KEY'),
)

def authenticate_client():
    ta_credential = AzureKeyCredential(os.getenv('TA_CREDENTIAL'))
    text_analytics_client = TextAnalyticsClient(
            endpoint=os.getenv('TA_ENDPOINT'),
            credential=ta_credential)
    return text_analytics_client

class FormData(db.Model):
    __tablename__ = 'form_data'

    id = db.Column(db.Integer, primary_key=True)
    gender = db.Column(db.CHAR(1))
    age = db.Column(db.String(255))
    todays_event = db.Column(db.Text)
    memorable_thing = db.Column(db.Text)
    one_word = db.Column(db.Text)

class AiDiaries(db.Model):
    __tablename__ = 'ai_diaries'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255))
    diaries = db.Column(db.Text)
    image_name = db.Column(db.String(255))
    ana_result = db.Column(db.String(255))
    ana_positive = db.Column(db.Float)
    ana_neutral = db.Column(db.Float)
    ana_negative = db.Column(db.Float)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

class Blog(db.Model):
    __tablename__ = 'blogs'
    id = db.Column(db.Integer,primary_key=True)
    name = db.Column(db.String(255),nullable=False)
    url = db.Column(db.String(255),nullable=False)
    image_name = db.Column(db.String(255),nullable=False)
    description = db.Column(db.Text,nullable=False)

@app.route('/api', methods=['POST'])
def api():

    FormData.query.delete()
    db.session.commit()

    data = request.get_json()

    form_data = FormData(
        gender=data.get('gender'),
        age=data.get('age'),
        todays_event=data.get('todaysEvent'),
        memorable_thing=data.get('memorableThing'),
        one_word=data.get('oneWord')
    )

    db.session.add(form_data)
    db.session.commit()

    return {"message": "Data inserted successfully"}, 201


@app.route('/api/generalai')
def generalai():

    form_data = FormData.query.first()
    
    if form_data:

        today = datetime.date.today()


        if form_data.gender == '0':
            gender = "男性"
        else:
            gender = "女性"

        age = form_data.age
        todays_event = form_data.todays_event
        memorable_thing = form_data.memorable_thing
        one_word = form_data.one_word

        user = f"{age}代の{gender}"

        response = client.chat.completions.create(
            model="ketyia1111",
            messages = 
            [{"role":"system","content":f"あなたは日記クリエータです。{user}になりきり、日記を300字以内で作成してください"},
            {"role":"user","content":f"{today}、{todays_event}がありました。特に印象に残っていることは{memorable_thing}。今日一日は{one_word}だった。"
            }],
        )

        form_data = FormData.query.filter_by(todays_event=todays_event).first()
        
        if form_data:
            db.session.delete(form_data)
            db.session.commit()

        response_message = response.choices[0].message.content

        data = {"example": f"{response_message}"}
        return jsonify(data),200

    else:

        return abort(404, description="Resource not found")


@app.route('/api/generalai/complete', methods=['POST'])
def generalai_complete():
    data = request.get_json()
    name = data.get('name')
    text = data.get('text')

    textana_client = authenticate_client()

    text_response = textana_client.analyze_sentiment(documents=[f"{text}"])[0]

    hashids = Hashids(
        min_length=15,  
        alphabet='abcdefghijklmnopqrstuvwxyz0123456789' 
    )

    # 現在の日付と時間を取得
    now = datetime.datetime.now()

    # 日付と時間を指定された形式に変換
    formatted_date = now.strftime("%Y%m%d%H%M%S")

    # 文字列を数値に変換
    numeric_date = int(formatted_date)

    image_name = hashids.encode(numeric_date)

    image_name += '.png'
    result = images.images.generate(
        model="dall-e-3",
        prompt=text,
        n=1
    )
    image_url = json.loads(result.model_dump_json())['data'][0]['url']

    # BlobServiceClientオブジェクトを作成します
    blob_service_client = BlobServiceClient.from_connection_string(os.getenv('BLOB_CONNECTION_STRING'))

    # BlobClientオブジェクトを作成します
    blob_client = blob_service_client.get_blob_client("images", image_name)

    # 指定されたURLから画像を取得します
    response = requests.get(image_url, stream=True)
    response.raise_for_status()

    # Blobにアップロードします
    blob_client.upload_blob(response.raw, blob_type="BlockBlob", length=None)
    
    ai_diaries = AiDiaries(
        username=name,
        diaries=text,
        image_name=image_name,
        ana_result=text_response.sentiment,
        ana_positive=float("{:.2f}".format(text_response.confidence_scores.positive)),
        ana_neutral=float("{:.2f}".format(text_response.confidence_scores.neutral)),
        ana_negative=float("{:.2f}".format(text_response.confidence_scores.negative))
    )

    db.session.add(ai_diaries)
    db.session.commit()

    data = {"example": f"{text_response.sentiment}"}

    return jsonify(data),200

@app.route('/api/list/<int:page_num>')
def items(page_num):
    items = AiDiaries.query.order_by(AiDiaries.id.desc()).paginate(per_page=10, page=page_num, error_out=True)
    items_list = [{c.name: getattr(item, c.name) for c in AiDiaries.__table__.columns} for item in items.items]
    return jsonify({'items': items_list, 'page': page_num})

@app.route('/blogs', methods=['GET'])
def get_blogs():
    blogs = Blog.query.all()
    output = []
    for blog in blogs:
        blog_data = {
            'name': blog.name,
            'url': blog.url,
            'image_name': blog.image_name,
            'description': blog.description
        }
        output.append(blog_data)
    return jsonify({'blogs': output})



if __name__ == "__main__":
    app.run()
