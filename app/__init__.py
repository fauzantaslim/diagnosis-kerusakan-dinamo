from flask import Flask
from config import Config
from app.models import db
from flask_migrate import Migrate
from flask_cors import CORS


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize extensions
    CORS(app, supports_credentials=True)
    db.init_app(app)
    Migrate(app, db)

    # Register blueprints
    from app.routes.auth import auth_bp
    from app.routes.dashboard import dashboard_bp
    from app.routes.diagnosis import diagnosis_bp
    from app.routes.history import history_bp
    from app.routes.dataset import dataset_bp
    from app.routes.rf import rf_bp
    from app.routes.evaluation import evaluation_bp

    app.register_blueprint(auth_bp,       url_prefix='/auth')
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(diagnosis_bp,  url_prefix='/diagnosis')
    app.register_blueprint(history_bp,    url_prefix='/history')
    app.register_blueprint(dataset_bp,    url_prefix='/api/dataset')
    app.register_blueprint(rf_bp,         url_prefix='/api/rf')
    app.register_blueprint(evaluation_bp, url_prefix='/api/evaluation')

    return app
