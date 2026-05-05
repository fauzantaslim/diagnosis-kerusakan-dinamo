import os
from flask import Flask
from dotenv import load_dotenv
from models import db, User
from flask_login import LoginManager
from routes import main_bp

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default_secret_key')

# Database configuration
db_user = os.environ.get('DB_USER', 'root')
db_password = os.environ.get('DB_PASSWORD', '')
db_host = os.environ.get('DB_HOST', 'localhost')
db_port = os.environ.get('DB_PORT', '3306')
db_name = os.environ.get('DB_NAME', 'dinamo_db')

app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)

login_manager = LoginManager()
login_manager.login_view = 'main_bp.login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Register blueprints
app.register_blueprint(main_bp)

# Create tables before first request
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)
