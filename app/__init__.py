from flask import Flask
from config import Config
from app.models import db
from flask_login import LoginManager

login_manager = LoginManager()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'auth_bp.login'

    # User loader
    from app.models.user import User
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # Register blueprints
    from app.routes.auth import auth_bp
    from app.routes.dashboard import dashboard_bp
    from app.routes.diagnosis import diagnosis_bp
    from app.routes.dataset import dataset_bp

    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(diagnosis_bp, url_prefix='/diagnosis')
    app.register_blueprint(dataset_bp, url_prefix='/dataset')

    return app
