from flask import Flask, request, render_template, send_from_directory
from flask import Flask, session, redirect, url_for
import os
import random
from PIL import Image
from PIL.ExifTags import TAGS
from PIL import ExifTags

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

def getEXIF(img_path):
    
    img = Image.open(img_path)
    info_dict = {
        "Image Size": img.size,
        "Image Height": img.height,
        "Image Width": img.width,
        "Image Format": img.format,
        "Image Mode": img.mode,
        "Image is Animated": getattr(img, "is_animated", False),
        "Frames in Image": getattr(img, "n_frames", 1)
    }
    
    exif = { ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS }
    try:
        # iterating over all EXIF data fields
        for tag_id in exif:
            # get the tag name, instead of human unreadable tag id
            tag = TAGS.get(tag_id, tag_id)
            data = exif.get(tag_id)
            # decode bytes 
            if isinstance(data, bytes):
                data = data.decode()
            info_dict.update({tag: data})
    except:
        print("Error while reading image metadata...")
        print(info_dict)
    return info_dict

#index web app
@app.route("/")
def index():
    #Controllo sessione dell'utente
    if not session.get('user') is None:
        print("Sessione già creata per l'utente")
    else:
        print("Nuovo Utente, creazione sessione")
        session["user"] = random.random()
    return render_template("upload3.html")

#Metodo richiamato quando si carica un'immagine sul server
@app.route("/", methods=["POST"])
def upload():
    if request.method == 'POST':
        target = os.path.join(APP_ROOT, 'images/')
        #Creazione cartella ./images se non presente
        if not os.path.isdir(target):
            os.mkdir(target)

        #Salvataggio immagine caricata dall'utente nella cartella ./images
        for upload in request.files.getlist("file"):
            print("Il file caricato è {}".format(upload.filename))
            filename = upload.filename
            
            destination = "".join([target, filename])
            #Salvataggio immagine nella cartella ./images
            upload.save(destination)
            metadata = getEXIF(destination)
            session["filename"] = filename
            session["metadata"] = metadata
    
    #Extraction EXIF and compute the noiseprint of the image and then find the name of the model
    return redirect(url_for("results"))

#Funzione che invia dal server al client le immagini simili estratte
@app.route('/results')
def results():
    return render_template("results.html", filename=session.get("filename"), metadata=session.get("metadata"))

@app.route('/results/<filename>')
def send_image(filename):
    print("entro")
    source = "images"
    return send_from_directory(source, filename)

if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(port=4555, debug=True, use_reloader=True)