from flask import Flask, request, jsonify
import numpy as np
import logging
import joblib

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the pre-trained model
try:
    with open("fpl_xgbr_model.pkl", "rb") as model_file:
        model = joblib.load("fpl_xgbr_model.pkl")
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")


@app.route("/predict_points", methods=["POST"])
def predict_points():
    logging.info("Received a request to /predict_points")
    data = request.get_json()
    logging.info(f"Request data: {data}")

    # Required input fields
    required_fields = [
        "player_value",
        "is_home_game",
        "previous_assists",
        "previous_bonus",
        "previous_clean_sheets",
        "previous_goals_conceded",
        "previous_goals_scored",
        "previous_ict_index",
        "previous_minutes",
        "previous_red_cards",
        "previous_saves",
        "previous_total_points",
        "previous_yellow_cards",
        "player_form",
        "team_form",
        "player_position_DEF",
        "player_position_FWD",
        "player_position_GKP",
        "player_position_MID",
        "player_team_name_Arsenal",
        "player_team_name_Aston Villa",
        "player_team_name_Bournemouth",
        "player_team_name_Brentford",
        "player_team_name_Brighton",
        "player_team_name_Burnley",
        "player_team_name_Cardiff",
        "player_team_name_Chelsea",
        "player_team_name_Crystal Palace",
        "player_team_name_Everton",
        "player_team_name_Fulham",
        "player_team_name_Huddersfield",
        "player_team_name_Hull",
        "player_team_name_Leeds",
        "player_team_name_Leicester",
        "player_team_name_Liverpool",
        "player_team_name_Luton",
        "player_team_name_Man City",
        "player_team_name_Man Utd",
        "player_team_name_Middlesbrough",
        "player_team_name_Newcastle",
        "player_team_name_Norwich",
        "player_team_name_Nott'm Forest",
        "player_team_name_Sheffield Utd",
        "player_team_name_Southampton",
        "player_team_name_Spurs",
        "player_team_name_Stoke",
        "player_team_name_Sunderland",
        "player_team_name_Swansea",
        "player_team_name_Watford",
        "player_team_name_West Brom",
        "player_team_name_West Ham",
        "player_team_name_Wolves",
        "opposition_team_name_Arsenal",
        "opposition_team_name_Aston Villa",
        "opposition_team_name_Bournemouth",
        "opposition_team_name_Brentford",
        "opposition_team_name_Brighton",
        "opposition_team_name_Burnley",
        "opposition_team_name_Cardiff",
        "opposition_team_name_Chelsea",
        "opposition_team_name_Crystal Palace",
        "opposition_team_name_Everton",
        "opposition_team_name_Fulham",
        "opposition_team_name_Huddersfield",
        "opposition_team_name_Hull",
        "opposition_team_name_Leeds",
        "opposition_team_name_Leicester",
        "opposition_team_name_Liverpool",
        "opposition_team_name_Luton",
        "opposition_team_name_Man City",
        "opposition_team_name_Man Utd",
        "opposition_team_name_Middlesbrough",
        "opposition_team_name_Newcastle",
        "opposition_team_name_Norwich",
        "opposition_team_name_Nott'm Forest",
        "opposition_team_name_Sheffield Utd",
        "opposition_team_name_Southampton",
        "opposition_team_name_Spurs",
        "opposition_team_name_Stoke",
        "opposition_team_name_Sunderland",
        "opposition_team_name_Swansea",
        "opposition_team_name_Watford",
        "opposition_team_name_West Brom",
        "opposition_team_name_West Ham",
        "opposition_team_name_Wolves",
    ]

    # Check if all required fields are present
    if not data:
        logging.error("No JSON payload received.")
        return jsonify({"error": "No JSON payload received."}), 400

    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        logging.error(f"Missing fields: {missing_fields}")
        return jsonify({"error": f'Missing fields: {", ".join(missing_fields)}.'}), 400

    try:
        # Extract the input features in the correct order
        input_features = [
            float(data["player_value"]),
            float(data["is_home_game"]),
            float(data["previous_assists"]),
            float(data["previous_bonus"]),
            float(data["previous_clean_sheets"]),
            float(data["previous_goals_conceded"]),
            float(data["previous_goals_scored"]),
            float(data["previous_ict_index"]),
            float(data["previous_minutes"]),
            float(data["previous_red_cards"]),
            float(data["previous_saves"]),
            float(data["previous_total_points"]),
            float(data["previous_yellow_cards"]),
            float(data["player_form"]),
            float(data["team_form"]),
            float(data["player_position_DEF"]),
            float(data["player_position_FWD"]),
            float(data["player_position_GKP"]),
            float(data["player_position_MID"]),
            float(data["player_team_name_Arsenal"]),
            float(data["player_team_name_Aston Villa"]),
            float(data["player_team_name_Bournemouth"]),
            float(data["player_team_name_Brentford"]),
            float(data["player_team_name_Brighton"]),
            float(data["player_team_name_Burnley"]),
            float(data["player_team_name_Cardiff"]),
            float(data["player_team_name_Chelsea"]),
            float(data["player_team_name_Crystal Palace"]),
            float(data["player_team_name_Everton"]),
            float(data["player_team_name_Fulham"]),
            float(data["player_team_name_Huddersfield"]),
            float(data["player_team_name_Hull"]),
            float(data["player_team_name_Leeds"]),
            float(data["player_team_name_Leicester"]),
            float(data["player_team_name_Liverpool"]),
            float(data["player_team_name_Luton"]),
            float(data["player_team_name_Man City"]),
            float(data["player_team_name_Man Utd"]),
            float(data["player_team_name_Middlesbrough"]),
            float(data["player_team_name_Newcastle"]),
            float(data["player_team_name_Norwich"]),
            float(data["player_team_name_Nott'm Forest"]),
            float(data["player_team_name_Sheffield Utd"]),
            float(data["player_team_name_Southampton"]),
            float(data["player_team_name_Spurs"]),
            float(data["player_team_name_Stoke"]),
            float(data["player_team_name_Sunderland"]),
            float(data["player_team_name_Swansea"]),
            float(data["player_team_name_Watford"]),
            float(data["player_team_name_West Brom"]),
            float(data["player_team_name_West Ham"]),
            float(data["player_team_name_Wolves"]),
            float(data["opposition_team_name_Arsenal"]),
            float(data["opposition_team_name_Aston Villa"]),
            float(data["opposition_team_name_Bournemouth"]),
            float(data["opposition_team_name_Brentford"]),
            float(data["opposition_team_name_Brighton"]),
            float(data["opposition_team_name_Burnley"]),
            float(data["opposition_team_name_Cardiff"]),
            float(data["opposition_team_name_Chelsea"]),
            float(data["opposition_team_name_Crystal Palace"]),
            float(data["opposition_team_name_Everton"]),
            float(data["opposition_team_name_Fulham"]),
            float(data["opposition_team_name_Huddersfield"]),
            float(data["opposition_team_name_Hull"]),
            float(data["opposition_team_name_Leeds"]),
            float(data["opposition_team_name_Leicester"]),
            float(data["opposition_team_name_Liverpool"]),
            float(data["opposition_team_name_Luton"]),
            float(data["opposition_team_name_Man City"]),
            float(data["opposition_team_name_Man Utd"]),
            float(data["opposition_team_name_Middlesbrough"]),
            float(data["opposition_team_name_Newcastle"]),
            float(data["opposition_team_name_Norwich"]),
            float(data["opposition_team_name_Nott'm Forest"]),
            float(data["opposition_team_name_Sheffield Utd"]),
            float(data["opposition_team_name_Southampton"]),
            float(data["opposition_team_name_Spurs"]),
            float(data["opposition_team_name_Stoke"]),
            float(data["opposition_team_name_Sunderland"]),
            float(data["opposition_team_name_Swansea"]),
            float(data["opposition_team_name_Watford"]),
            float(data["opposition_team_name_West Brom"]),
            float(data["opposition_team_name_West Ham"]),
            float(data["opposition_team_name_Wolves"]),
        ]

        # Convert to numpy array and reshape for prediction
        input_array = np.array(input_features).reshape(1, -1)

        # Make prediction
        predicted_points = model.predict(input_array)[0]
        predicted_points = float(predicted_points)

        return jsonify({"predicted_points": round(predicted_points, 2)})

    except ValueError as ve:
        logging.error(f"ValueError: {ve}")
        return (
            jsonify({"error": "Invalid input. Ensure all fields are numerical."}),
            400,
        )
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500


if __name__ == "__main__":
    app.run(debug=True)
