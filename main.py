from flask import Flask
from file_upload import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)