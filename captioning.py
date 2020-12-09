from flask import Flask,render_template, request
from predictor_api import make_prediction
from werkzeug.utils import secure_filename
from gpt3_functions import new_story_with_caption,continue_story_with_caption,continue_story_without_caption,continue_story_with_text
import os


app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

def upload_image(f):
	upload_path = './static/'
	#f = request.files['image-input']
	filename ='download.jpeg'
	path = upload_path+filename
	f.save(os.path.join(upload_path, filename))	



@app.route('/')
def hello():
	return render_template('index.html',image=None, result=None,summary=None)

@app.route('/predict/<int:choice>', methods=["GET", "POST"])
def predict(choice):
	if request.method == 'POST':
		f = None
		caption= None
		summary = None

		if choice==1 or choice==2: # new story on image upload
			f = request.files['image-input']
			upload_image(f)
			caption = make_prediction()
			#if choice ==1:
			if request.form['image_button'] == 'new-image':
				summary=new_story_with_caption(caption)
			else:
				summary=continue_story_with_caption(caption)
		elif choice ==3: # continue story without image
			caption=" "
			summary=continue_story_without_caption()
		elif choice ==4: # continue story with user text
			caption = request.form['inputtext']
			summary=continue_story_with_text(caption)
	return render_template('index.html', image=f, result=caption,summary=summary)


@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

if __name__ == '__main__':
   app.run(debug = True)
