{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "15eb0962",
   "metadata": {},
   "outputs": [],
   "source": [
    "# librerías\n",
    "# Importaciones necesarias\n",
    "from flask import Flask\n",
    "from flask_restx import Api, Resource, fields\n",
    "from flask_cors import CORS\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import xgboost as xgb\n",
    "from xgboost import XGBRegressor\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "e2ad73fa",
   "metadata": {},
   "outputs": [],
   "source": [
    "# cargar datos (se tiene en .csv en local)\n",
    "df_train=pd.read_csv(\"dataTrain_carListings/dataTrain_carListings.csv\")\n",
    "# data test tiene una columna llamada ID, que solamente es el orden de numeros\n",
    "df_test=pd.read_csv(\"dataTest_carListings/dataTest_carListings.csv\", index_col=0)\n",
    "# Cargar datos reales\n",
    "# df_real = pd.read_csv(\"true_car_listings.csv\", on_bad_lines='skip')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "1716f7f8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# eliminar columnas adicionales\n",
    "# df_real.drop([\"City\", \"Vin\"], axis=1, inplace=True)\n",
    "# df_real.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "4cb27ace",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Codificar variables categóricas\n",
    "categorical_columns = ['State', 'Make', 'Model']\n",
    "df_train = pd.get_dummies(df_train, columns=categorical_columns).astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "96e7ff96",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Tipo de X_train: <class 'pandas.core.frame.DataFrame'>\n",
      "Tipo de X_test: <class 'pandas.core.frame.DataFrame'>\n",
      "Tipo de y_train: <class 'pandas.core.series.Series'>\n",
      "Tipo de y_test: <class 'pandas.core.series.Series'>\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "data = df_train.drop(['Price'], axis=1)\n",
    "target = df_train['Price']\n",
    "\n",
    "# Dividir los datos en conjuntos de entrenamiento y prueba\n",
    "X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=42)\n",
    "\n",
    "# Convertir los arrays resultantes en DataFrames de pandas\n",
    "X_train = pd.DataFrame(X_train, columns=data.columns)\n",
    "X_test = pd.DataFrame(X_test, columns=data.columns)\n",
    "y_train = pd.Series(y_train, name='Price')\n",
    "y_test = pd.Series(y_test, name='Price')\n",
    "\n",
    "# Verifica los tipos de datos de las estructuras resultantes\n",
    "print(\"Tipo de X_train:\", type(X_train))\n",
    "print(\"Tipo de X_test:\", type(X_test))\n",
    "print(\"Tipo de y_train:\", type(y_train))\n",
    "print(\"Tipo de y_test:\", type(y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "b8903b5c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['xgb_model.pkl']"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import joblib\n",
    "\n",
    "# Configuración y entrenamiento del modelo\n",
    "xgb_2 = XGBRegressor(\n",
    "    reg_lambda=0.16062324818666773,\n",
    "    alpha=0.6139702492127704,\n",
    "    subsample=0.9702623758315497,\n",
    "    colsample_bytree=0.539127268885431,\n",
    "    n_estimators=941,\n",
    "    max_depth=9,\n",
    "    min_child_weight=8,\n",
    "    learning_rate=0.15277968834080027,\n",
    "    gamma=0.410872106042996,\n",
    "    random_state=659\n",
    ")\n",
    "\n",
    "xgb_2.fit(X_train, y_train)\n",
    "\n",
    "# Guardar el modelo entrenado\n",
    "joblib.dump(xgb_2, 'xgb_model.pkl')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "17aefa39",
   "metadata": {},
   "outputs": [],
   "source": [
    "from flask import Flask\n",
    "from flask_restx import Api, Resource, fields\n",
    "from flask_cors import CORS\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import joblib\n",
    "from werkzeug.exceptions import BadRequest\n",
    "\n",
    "app = Flask(__name__)\n",
    "CORS(app)\n",
    "api = Api(app, version='1.0', title='Vehicle Price Prediction API',\n",
    "          description='API for predicting vehicle prices based on their features')\n",
    "\n",
    "ns = api.namespace('price', description='Price Estimator')\n",
    "\n",
    "parser = api.parser()\n",
    "parser.add_argument('Year', type=int, required=True, help='Year of the vehicle', location='args')\n",
    "parser.add_argument('Mileage', type=int, required=True, help='Mileage of the vehicle', location='args')\n",
    "parser.add_argument('State', type=str, required=True, help='State where the vehicle is sold', location='args')\n",
    "parser.add_argument('Make', type=str, required=True, help='Make of the vehicle', location='args')\n",
    "parser.add_argument('Model', type=str, required=True, help='Model of the vehicle', location='args')\n",
    "\n",
    "resource_fields = api.model('Resource', {\n",
    "    'predicted_price': fields.Float,\n",
    "})\n",
    "\n",
    "try:\n",
    "    model = joblib.load('xgb_model.pkl')\n",
    "except Exception as e:\n",
    "    print(f\"Error loading model: {e}\")\n",
    "    raise\n",
    "\n",
    "@ns.route('/')\n",
    "class PricePredictor(Resource):\n",
    "    @api.doc(parser=parser)\n",
    "    @api.marshal_with(resource_fields)\n",
    "    def get(self):\n",
    "        args = parser.parse_args()\n",
    "        # Crear un DataFrame con una fila de ceros con las columnas dummies requeridas\n",
    "        input_data = pd.DataFrame(0, index=np.arange(1), columns=model.get_booster().feature_names)\n",
    "        \n",
    "        # Actualizar los valores de Year y Mileage\n",
    "        input_data.at[0, 'Year'] = args['Year']\n",
    "        input_data.at[0, 'Mileage'] = args['Mileage']\n",
    "        \n",
    "        # Actualizar las columnas dummies para State, Make, Model\n",
    "        input_data.at[0, f'State_{args[\"State\"].strip()}'] = 1\n",
    "        input_data.at[0, f'Make_{args[\"Make\"]}'] = 1\n",
    "        input_data.at[0, f'Model_{args[\"Model\"]}'] = 1\n",
    "        \n",
    "        # Asegúrate de que las columnas están en el mismo orden que las características del modelo\n",
    "        input_data = input_data.reindex(columns=model.get_booster().feature_names, fill_value=0)\n",
    "        \n",
    "        # Predecir el precio\n",
    "        predicted_price = model.predict(input_data)[0]\n",
    "\n",
    "        return {'predicted_price': predicted_price}, 200"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": none,
   "id": "66ccb4c6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app '__main__'\n",
      " * Debug mode: on\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\n",
      " * Running on all addresses (0.0.0.0)\n",
      " * Running on http://127.0.0.1:5000\n",
      " * Running on http://192.168.0.11:5000\n",
      "Press CTRL+C to quit\n",
      "127.0.0.1 - - [25/Apr/2024 20:00:23] \"GET / HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [25/Apr/2024 20:00:23] \"GET /swaggerui/droid-sans.css HTTP/1.1\" 304 -\n",
      "127.0.0.1 - - [25/Apr/2024 20:00:23] \"GET /swaggerui/swagger-ui-bundle.js HTTP/1.1\" 304 -\n",
      "127.0.0.1 - - [25/Apr/2024 20:00:23] \"GET /swaggerui/swagger-ui.css HTTP/1.1\" 304 -\n",
      "127.0.0.1 - - [25/Apr/2024 20:00:23] \"GET /swaggerui/swagger-ui-standalone-preset.js HTTP/1.1\" 304 -\n",
      "127.0.0.1 - - [25/Apr/2024 20:00:23] \"GET /swagger.json HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [25/Apr/2024 20:02:53] \"GET /price/?Year=2015&Mileage=18681&State=MO&Make=Buick&Model=EncoreConvenience HTTP/1.1\" 200 -\n"
     ]
    }
   ],
   "source": [
    "# Ejecución de la aplicación que disponibiliza el modelo de manera local en el puerto 5000\n",
    "# app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)\n",
    "if __name__ == '__main__':\n",
    "    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}