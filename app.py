from flask import Flask, flash, request, redirect, url_for, render_template,send_file,session,Response,send_from_directory
import urllib.request
import os
from werkzeug.utils import secure_filename
from shutil import copyfile
import main3
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io


app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('D:/My Projects/BITHACK/static', 'images')
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

prediction = "0"
a=0

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/upload', methods=['POST'])
def covnvert():
    
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            global prediction
            # prediction=main3.bounding_box2(filename)
            prediction=main3.input(filename)
              
            print("app",prediction)
            # generate_pdf()
        return   render_template('index.html', result=prediction)


    

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import os
    output_directory = 'D:/My Projects/BITHACK'
    c = canvas.Canvas(os.path.join(output_directory, "generated.pdf"), pagesize=letter)
    pdfmetrics.registerFont(TTFont('TamilFont', 'D:/My Projects/BITHACK/latha.ttf'))
    c.setFont('TamilFont', 12)
    tamil_text = prediction
    c.drawString(100, 600, tamil_text)
    c.save() 
    print("PDF created successfully.")
    return send_from_directory("D:/My Projects/BITHACK", "generated.pdf", as_attachment=True)
@app.route('/translate', methods=['POST'])
def translate():
    from googletrans import Translator
    text = prediction
    target_lang = 'en'
    translator = Translator()
    translated_text = translator.translate(text, dest=target_lang).text
    print("translate")
    # print(prediction)
    print(text)
    return render_template('index.html',translated_text=translated_text, result=prediction)

@app.route('/text_to_audio',methods=['GET'])
def text_to_audio():
    from gtts import gTTS
    text = prediction
    language = 'ta' 
    tts = gTTS(text, lang=language, slow=False)  
    audio_path = "D:/My Projects/BITHACK/static/audio/audio.mp3" 
    tts.save(audio_path)
    return send_file(audio_path, mimetype='audio/mpeg')

@app.route('/download_text_to_audio',methods=['GET'])
def download_text_to_audio():
    audio_path = "D:/My Projects/BITHACK/static/audio/audio.mp3" 
    return send_file(audio_path, as_attachment=True, download_name='output.mp3')


if __name__ == "__main__":
    app.run(threaded=True)