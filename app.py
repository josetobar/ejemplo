from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

# Lista de las 19 columnas/características esperadas (DEBEN coincidir con el orden del entrenamiento)
COLUMNAS_ESPERADAS = [
    'Hours_Studied', 'Attendance', 'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 
    'Sleep_Hours', 'Previous_Scores', 'Motivation_Level', 'Internet_Access', 'Tutoring_Sessions', 
    'Family_Income', 'Teacher_Quality', 'School_Type', 'Peer_Influence', 'Physical_Activity', 
    'Learning_Disabilities', 'Parental_Education_Level', 'Distance_from_Home', 'Gender'
]

app = Flask(__name__)
# Habilitar CORS para permitir que tu HTML (frontend) haga peticiones
CORS(app) 

# Cargar el modelo guardado al iniciar la API
try:
    with open('modelo_regresion.pkl', 'rb') as file:
        modelo_cargado = pickle.load(file)
    print("Modelo de Regresión Lineal cargado exitosamente.")
except FileNotFoundError:
    print("ERROR: No se encontró 'modelo_regresion.pkl'. Asegúrate de guardarlo primero.")
    modelo_cargado = None

@app.route('/predict', methods=['POST'])
def predict():
    if not modelo_cargado:
        return jsonify({"error": "Modelo no cargado"}), 500
        
    try:
        # 1. Obtener los datos del cuerpo de la petición POST (JSON)
        datos = request.get_json(force=True)
        
        # 2. Verificar que todos los 19 datos estén presentes y en orden
        # La solicitud debe enviar los valores en el orden de COLUMNAS_ESPERADAS.
        
        # Creamos una lista de valores numéricos en el orden correcto
        valores = [datos[col] for col in COLUMNAS_ESPERADAS]
        
        # 3. Convertir a un formato que Scikit-learn pueda entender (Array de NumPy)
        # Reshape(-1, 19) asegura que tenga la forma (1, 19)
        entrada = np.array(valores).reshape(1, -1)
        
        # 4. Realizar la predicción
        prediccion = modelo_cargado.predict(entrada)
        
        # 5. Formatear la respuesta
        nota_predicha = round(prediccion[0], 2)
        
        return jsonify({
            'nota_predicha': nota_predicha,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({"error": f"Error al procesar la petición o la predicción: {str(e)}"}), 400

if __name__ == '__main__':
    # Ejecutar la API en el puerto 5000
    print("Iniciando la API en http://127.0.0.1:5000")
    app.run(debug=True)