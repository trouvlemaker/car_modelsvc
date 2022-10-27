# Run a test server
from flask import Flask, render_template
# from flask_sqlalchemy import SQLAlchemy, BaseQuery
# from sqlalchemy.exc import OperationalError
from time import sleep
import urllib3


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Define the WSGI application object
app = Flask(__name__)

# Configurations
# app.config.from_object('app.config')
# session_options = dict(
#     bind=None,
#     autoflush=True,
#     # autocommit=True,
#     expire_on_commit=True
# )
# db = SQLAlchemy(app, session_options=session_options, query_class=RetryingQuery)


# Import a module / component using its blueprint handler variable (mod_auth)
from app.mod_service.controllers import mod_service as service


# Sample HTTP error handling
@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404


# Register blueprint(s)
app.register_blueprint(service)

# db.create_all()

