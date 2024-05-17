# safers_config.py
# reads configuration values from file .env
# uses python-dotenv package

from dotenv import dotenv_values

conf = dotenv_values(".env")  

OAUTH_URL = conf["OAUTH_URL"]
CKAN_URL = conf["CKAN_URL"]
OAUTH_API_KEY = conf["OAUTH_API_KEY"]
OAUTH_APP_ID = conf["OAUTH_APP_ID"]
OAUTH_USER = conf["OAUTH_USER"]
OAUTH_PWD = conf["OAUTH_PWD"]

# Keys to access S3 data storage for input data
S3_KEY = conf["S3_KEY"]
S3_SECRET = conf["S3_SECRET"]