"""
app/app.py — Flask application factory.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    # Config
    app.config["SECRET_KEY"]          = os.urandom(32)
    app.config["SQLALCHEMY_DATABASE_URI"]       = config.DATABASE_URI
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    app.config["UPLOAD_FOLDER"]      = config.UPLOADS_DIR

    # Ensure upload and db directories exist
    os.makedirs(config.UPLOADS_DIR,  exist_ok=True)
    os.makedirs(config.DATABASE_DIR, exist_ok=True)

    # Init DB
    db.init_app(app)

    # Register blueprints
    from app.routes.main        import main_bp
    from app.routes.crop        import crop_bp
    from app.routes.disease     import disease_bp
    from app.routes.weather     import weather_bp
    from app.routes.history     import history_bp
    from app.routes.performance import performance_bp
    from app.routes.chat        import chat_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(crop_bp)
    app.register_blueprint(disease_bp)
    app.register_blueprint(weather_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(performance_bp)
    app.register_blueprint(chat_bp)

    with app.app_context():
        db.create_all()

    return app
