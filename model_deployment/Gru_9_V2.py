# Importaciones necesarias
from flask import Flask
from flask_restx import Api, Resource, fields
from flask_cors import CORS
import pandas as pd
from tensorflow.keras.models import load_model
from werkzeug.exceptions import BadRequest
from sklearn.model_selection import train_test_split
import warnings
import numpy as np
import joblib
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)
api = Api(app, version='1.0', title='Movie Genre Prediction API',
          description='API for predicting movie genres based on their title and plot')

ns = api.namespace('genre', description='Genre Predictor')

parser = api.parser()
parser.add_argument('title', type=str, required=True, help='Title of the movie', location='args')
parser.add_argument('plot', type=str, required=True, help='Plot of the movie', location='args')
parser.add_argument('year', type=int, required=True, help='Release year of the movie', location='args')

resource_fields = api.model('Resource', {
    'predicted_genres': fields.String,
})

try:
    # Cargar el modelo entrenado de Keras
    model = load_model('model.h5')
    # Cargar el vectorizador y el escalador
    vectorizer = joblib.load('vectorizer.pkl')
    scaler = joblib.load('scaler.pkl')
    # Cargar el LabelEncoder o MultiLabelBinarizer
    le = joblib.load('label_encoder.pkl')  # Asegúrate de tener este archivo y cargarlo aquí
except Exception as e:
    print(f"Error loading model or other components: {e}")
    raise

@ns.route('/')
class GenrePredictor(Resource):
    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        # Validar y preprocesar los datos de entrada
        title = args['title']
        plot = args['plot']
        year = args['year']
        
        if not title or not plot or not year:
            raise BadRequest('Title, plot, and year are required.')

        combined_text = title + " " + plot
        text_vectorized = vectorizer.transform([combined_text])
        year_scaled = scaler.transform([[year]])
        
        # Preparar la entrada final combinando texto y año
        input_data = np.hstack((text_vectorized.toarray(), year_scaled))
        
        # Predecir los géneros
        predictions = model.predict(input_data)
        predicted_genres = le.inverse_transform(np.round(predictions))  # Asumiendo que 'le' es el LabelEncoder usado
        
        return {'predicted_genres': ', '.join(predicted_genres[0])}, 200

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)

