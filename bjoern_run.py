import bjoern
from app import app
from app.common.utils import logger

# Run a test server
# app.run(host='0.0.0.0', port=8088, debug=True)
logger.info('Start application server...')
bjoern.run(app, '0.0.0.0', 8088)