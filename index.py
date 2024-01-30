from app import app
from app_callbacks import *

if __name__ == '__main__':
    app.run_server(host="localhost", debug=True)