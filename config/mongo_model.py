from flask_mongoengine import MongoEngine
import datetime

db = MongoEngine()

class Benchmarks(db.Document):
    language = db.StringField(required=True,max_length=30)
    type = db.StringField(max_length=20)
    user_filename = db.StringField(max_length=100)
    db_filename = db.StringField(max_length=100)
    version = db.FloatField(default=1)
    created_at = db.DateTimeField(default=datetime.datetime.now)   
    created_by = db.StringField(max_length=60)
    path = db.StringField(max_length=200)    