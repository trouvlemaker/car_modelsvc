import os
from omegaconf import OmegaConf


application_path = os.path.dirname(os.path.abspath(__file__))
config_path = f"{application_path}/configs"
config_name = 'app_config.yaml'
cfg = OmegaConf.load(f"{config_path}/{config_name}")
config = OmegaConf.to_container(cfg)
print(f'configuration file loaded {application_path}')

# Application threads. A common general assumption is
# using 2 per available processor cores - to handle
# incoming requests using one and performing background
# operations using the other.
THREADS_PER_PAGE = 2

# Enable protection agains *Cross-site Request Forgery (CSRF)*
CSRF_ENABLED     = True

# Use a secure, unique and absolutely secret key for
# signing the data.
CSRF_SESSION_KEY = "secret"

# Secret key for signing cookies
SECRET_KEY = "secret"