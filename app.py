from flask import Flask,render_template,request
import numpy as np
import pickle

# Load ML model
model = pickle.load(open('model.pkl', 'rb')) 

app = Flask(__name__)

@app.route('/')
def index():
	return render_template("klasifikasi jamu.html")

@app.route('/klasifikasi')
def klasifikasi():
	return render_template("test-ui.html")


@app.route('/',methods=["POST"])
def predict():
	if request.method == 'POST':
		TGS_813 = request.form['TGS 813']
		TGS_2611 = request.form['TGS 2611']
		MQ136 = request.form['MQ136']

		# Clean the data by convert from unicode to float
		sample_data = [TGS_813,TGS_2611,MQ136]
		clean_data = [float(i) for i in sample_data]

		# Reshape the Data as a Sample not Individual Features
		ex1 = np.array(clean_data).reshape(1,-1)
		prediction = model.predict(ex1)

	return render_template('klasifikasi jamu.html', 
		TGS_813=TGS_813,
		TGS_2611=TGS_2611,
		MQ136=MQ136,
		clean_data=clean_data,
		prediction=prediction)

if __name__ == '__main__':
#Run the application
    #Develop Mode
    #app.run()
    
    #///////////
    #Debug Mode
    app.run(debug=True, use_reloader=True)