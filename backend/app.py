from copy import deepcopy
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

SEEDED_USERS = {
    "1": {"id": "1", "first_name": "Ava", "user_group": 11},
    "2": {"id": "2", "first_name": "Ben", "user_group": 22},
    "3": {"id": "3", "first_name": "Chloe", "user_group": 33},
    "4": {"id": "4", "first_name": "Diego", "user_group": 44},
    "5": {"id": "5", "first_name": "Ella", "user_group": 55},
}

MODEL_PATH = Path(__file__).resolve().parent / "src" / "random_forest_model.pkl"
PREDICTION_COLUMNS = [
    "city",
    "province",
    "latitude",
    "longitude",
    "lease_term",
    "type",
    "beds",
    "baths",
    "sq_feet",
    "furnishing",
    "smoking",
    "cats",
    "dogs",
]

app = Flask(__name__)
# For this lab, allow cross-origin requests from the React dev server.
# This broad setup keeps local development simple and is not standard
# production practice.
CORS(app)
users = deepcopy(SEEDED_USERS)
model = None


# TODO: Define these Flask routes with @app.route():
# - GET /users
#   Return 200 on success. The frontend still expects a JSON array,
#   so return list(users.values()) instead of the dict directly.
# - POST /users
#   Return 201 for a successful create, 400 for invalid input,
#   and 409 if the id already exists. Since users is a dict keyed by
#   id, use the id as the lookup key when checking for duplicates.
# - PUT /users/<user_id>
#   Return 200 for a successful update, 400 for invalid input,
#   and 404 if the user does not exist. Update the matching record
#   with users[user_id] = {...} after confirming the key exists.
# - DELETE /users/<user_id>
#   Return 200 for a successful delete and 404 if the user does not
#   exist. Delete with del users[user_id] after confirming the key
#   exists.
#   Exercise2
# - POST /predict_house_price


@app.route('/users', methods=['GET'])  
def get_users():
    return jsonify(list(users.values())), 200


@app.route('/users', methods=['POST'])
def create_user():
    data  = request.json 

    if not data or 'id' not in data or 'first_name' not in data or 'user_group' not in data: 
        return jsonify({"message": "Missing required fields."}), 400

    user_id = data['id']

    if user_id in users:
        return jsonify({"message": f"User {user_id} already exists."}), 409 

    users[user_id] = {
        "id": user_id,
        "first_name":  data['first_name'],
        "user_group": data['user_group']
    }

    return jsonify({
        "id": user_id,
        "first_name": data['first_name'],
        "user_group": data['user_group'],
        "message": f"Created user {user_id}."
    }), 201 


@app.route('/users/<user_id>', methods=['PUT'])
def update_user(user_id):
    if user_id not in users:
        return jsonify({"message": f"User {user_id} was not found."}), 404

    data = request.json

    if not data or 'first_name' not in data or 'user_group' not in data:
        return jsonify({"message": "Invalid request body."}), 400 

    users[user_id] = {
        "id": user_id,
        "first_name": data['first_name'],
        "user_group": data['user_group']
    }

    return jsonify({
        "id": user_id,
        "first_name": data['first_name'],
        "user_group": data['user_group'],
        "message": f"Updated user {user_id}." 
    }), 200


@app.route('/users/<user_id>', methods=['DELETE'])
def delete_user(user_id): 
    if user_id not in users: 
        return jsonify({"message": f"User {user_id} was not found."}), 404

    del users[user_id]

    return jsonify({"message": f"Deleted user {user_id}."}), 200


@app.route("/predict_house_price", methods=["POST"])
def predict_house_price():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"message": "Invalid input"}), 400

    required_fields = [
        "city",
        "province",
        "latitude",
        "longitude",
        "lease_term",
        "type",
        "beds",
        "baths",
        "sq_feet",
        "furnishing",
        "smoking",
        "pets",
    ]
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({"message": f"Missing fields: {', '.join(missing_fields)}"}), 400

    def parse_float(field_name):
        try:
            return float(data[field_name])
        except (TypeError, ValueError):
            raise ValueError(f"{field_name} must be a number.")

    try:
        model = joblib.load(MODEL_PATH)
        pets_allowed = bool(data["pets"])
        sample_data = [
            data["city"],
            data["province"],
            parse_float("latitude"),
            parse_float("longitude"),
            data["lease_term"],
            data["type"],
            parse_float("beds"),
            parse_float("baths"),
            parse_float("sq_feet"),
            data["furnishing"],
            data["smoking"],
            pets_allowed,
            pets_allowed,
        ]
        sample_df = pd.DataFrame([sample_data], columns=PREDICTION_COLUMNS)
        predicted_price = model.predict(sample_df)
        return jsonify({"predicted_price": float(predicted_price[0])}), 200
    except Exception as error:
        return jsonify({"message": str(error)}), 400


@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Backend is running"}), 200

if __name__ == "__main__":
    app.run(debug=True, port=5050) 
