'''
FLASK_APP=app.py FLASK_DEBUG=1 TEMPLATES_AUTO_RELOAD=1 flask run
or 
flask run
'''

import os, json
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import send_from_directory, jsonify

UPLOAD_FOLDER = './temp'
ALLOWED_EXTENSIONS = { 'png', 'jpg', 'jpeg', 'gif'}
DELETE_AFTER_PREDICTION=True
current_status= "running"
tokens={"id":101,"filename":""}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024*1024  # 16MB


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            token_dir=os.path.join(app.config['UPLOAD_FOLDER'], str(tokens['id'])) 
            os.mkdir(token_dir) if not os.path.isdir(token_dir) else None
            tokens['filename']=os.path.join(token_dir, filename)
            print (tokens['filename'])
            file.save(tokens['filename'])

            return redirect(url_for('predict', name=tokens['filename']))
    return '''
    <!doctype html>
    <title>Upload photo to be recognized!</title>
    <h1>Upload photo to be recognized!</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploads/<name>')
def download_file(name):
    token_dir=tokenize['id']
    os.mkdir(token_dir) if not os.path.isdir(token_dir) else None
    token_filename=os.path.join(token_dir,name)
    tokens['filename']=token_filename
    return send_from_directory(app.config["UPLOAD_FOLDER"], token_filename)

@app.route ('/tokenize/<token>')
def tokenize(token):
    tokens['id']=token
    return redirect(url_for('status'))

@app.route('/status')
def status():
    return jsonify ({"status":current_status, 'token':tokens['id']})

@app.route("/test")
def test():
    test_payload={"payload": tokens['filename']}
    return jsonify(test_payload)

@app.route('/predict')
def predict():
    path_file=tokens['filename']
    # todo: call function for prediction
    response_json=json.dumps( {'predicting':path_file})
    #os.unlink(path_file) if DELETE_AFTER_PREDICTION else None #delete token_dir on token expiry
    return response_json