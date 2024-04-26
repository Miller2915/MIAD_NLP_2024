# librerías
# Importaciones necesarias
from flask import Flask
from flask_restx import Api, Resource, fields
from flask_cors import CORS
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# cargar datos (se tiene en .csv en local)
df_train=pd.read_csv("dataTrain_carListings/dataTrain_carListings.csv")
# data test tiene una columna llamada ID, que solamente es el orden de numeros
df_test=pd.read_csv("dataTest_carListings/dataTest_carListings.csv", index_col=0)

# Codificar variables categóricas
categorical_columns = ['State', 'Make', 'Model']
# df_train = pd.get_dummies(df_train, columns=categorical_columns).astype(int)
df_train = pd.get_dummies(df_train, columns=categorical_columns).astype('int32')

import pandas as pd
from sklearn.model_selection import train_test_split
# data = df_train.drop(['Price'], axis=1)
# target = df_train['Price']

df_train_sampled = df_train.sample(frac=0.5, random_state=42)
data = df_train_sampled.drop(['Price'], axis=1)
target = df_train_sampled['Price']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=42)

# Convertir los arrays resultantes en DataFrames de pandas
X_train = pd.DataFrame(X_train, columns=data.columns)
X_test = pd.DataFrame(X_test, columns=data.columns)
y_train = pd.Series(y_train, name='Price')
y_test = pd.Series(y_test, name='Price')

# Verifica los tipos de datos de las estructuras resultantes
print("Tipo de X_train:", type(X_train))
print("Tipo de X_test:", type(X_test))
print("Tipo de y_train:", type(y_train))
print("Tipo de y_test:", type(y_test))

import joblib

# Configuración y entrenamiento del modelo
xgb_2 = XGBRegressor(
    reg_lambda=0.16062324818666773,
    alpha=0.6139702492127704,
    subsample=0.9702623758315497,
    colsample_bytree=0.539127268885431,
    n_estimators=941,
    max_depth=9,
    min_child_weight=8,
    learning_rate=0.15277968834080027,
    gamma=0.410872106042996,
    random_state=659
)

xgb_2.fit(X_train, y_train)

# Guardar el modelo entrenado
joblib.dump(xgb_2, 'xgb_model.pkl')

from flask import Flask
from flask_restx import Api, Resource, fields
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from werkzeug.exceptions import BadRequest

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