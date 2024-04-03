import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.abspath(os.getcwd()), '.env')
load_dotenv(dotenv_path)


class BaseConfig:
    app_path = os.path.abspath(os.getcwd())
    MFR_URL = os.environ.get('MFR_URL', None)
    MFR_AUTH_TOKEN = os.environ.get('MFR_AUTH_TOKEN', None)


settings = BaseConfig()
