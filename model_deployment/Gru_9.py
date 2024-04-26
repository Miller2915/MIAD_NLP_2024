# librerías
# Importaciones necesarias
from flask import Flask
from flask_restx import Api, Resource, fields
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from werkzeug.exceptions import BadRequest
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)
api = Api(app, version='1.0', title='Vehicle Price Prediction API',
          description='API for predicting vehicle prices based on their features')

ns = api.namespace('price', description='Price Estimator')

parser = api.parser()
parser.add_argument('Year', type=int, required=True, help='Year of the vehicle', location='args')
parser.add_argument('Mileage', type=int, required=True, help='Mileage of the vehicle', location='args')
parser.add_argument('State', type=str, required=True, help='State where the vehicle is sold', location='args')
parser.add_argument('Make', type=str, required=True, help='Make of the vehicle', location='args')
parser.add_argument('Model', type=str, required=True, help='Model of the vehicle', location='args')

resource_fields = api.model('Resource', {
    'predicted_price': fields.Float,
})

try:
    model = joblib.load('xgb_model.pkl')
except Exception as e:
    print(f"Error loading model: {e}")
    raise

@ns.route('/')
class PricePredictor(Resource):
    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        # Crear un DataFrame con una fila de ceros con las columnas dummies requeridas
        input_data = pd.DataFrame(0, index=np.arange(1), columns=model.get_booster().feature_names)
        
        # Actualizar los valores de Year y Mileage
        input_data.at[0, 'Year'] = args['Year']
        input_data.at[0, 'Mileage'] = args['Mileage']
        
        # Actualizar las columnas dummies para State, Make, Model
        input_data.at[0, f'State_{args["State"].strip()}'] = 1
        input_data.at[0, f'Make_{args["Make"]}'] = 1
        input_data.at[0, f'Model_{args["Model"]}'] = 1
        
        # Asegúrate de que las columnas están en el mismo orden que las características del modelo
        input_data = input_data.reindex(columns=model.get_booster().feature_names, fill_value=0)
        
        # Predecir el precio
        predicted_price = model.predict(input_data)[0]

        return {'predicted_price': predicted_price}, 200

# Ejecución de la aplicación que disponibiliza el modelo de manera local en el puerto 5000
# app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
