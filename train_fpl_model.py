#!/usr/bin/env python3

import os
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import logging

# Configure logging
logging.basicConfig(
    filename='fpl_model_training.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def create_code_to_name_mapping(seasons):
    id_to_code = pd.DataFrame()
    id_to_name = pd.DataFrame()
    code_to_name = pd.DataFrame()

    for season in seasons:
        # Load player raw data
        players_raw = pd.read_csv(
            f"historical_data/Fantasy-Premier-League/data/{season}/players_raw.csv",
            usecols=["id", "code", "team", "element_type"]
        )
        players_raw['season'] = season
        id_to_code = pd.concat([id_to_code, players_raw], ignore_index=True)
        id_to_code["id"] = id_to_code["id"].astype(str)
        id_to_code["code"] = id_to_code["code"].astype(str)

        # Load player ID list
        player_ids = pd.read_csv(
            f"historical_data/Fantasy-Premier-League/data/{season}/player_idlist.csv"
        )
        player_ids['season'] = season
        id_to_name = pd.concat([id_to_name, player_ids], ignore_index=True)
        id_to_name["name"] = id_to_name["first_name"] + " " + id_to_name["second_name"]
        id_to_name["id"] = id_to_name["id"].astype(str)

    # Merge to create code to name mapping
    merged_df = pd.merge(
        id_to_code,
        id_to_name[["id", "name", "season"]],
        on=["id", "season"],
        how="inner",
    )
    code_to_name = pd.concat([code_to_name, merged_df], ignore_index=True).drop_duplicates()
    return code_to_name

def clean_player_names(df):
    has_number = df["name"].str.split().str[-1].str.isnumeric()
    df["name"] = df["name"].where(
        ~has_number, df["name"].str.rsplit(" ", n=1).str[0]
    )
    df["name"] = df["name"].str.replace("_", " ")
    return df

def merge_team_names(df):
    master_teams = pd.read_csv("historical_data/Fantasy-Premier-League/data/master_team_list.csv")
    df = pd.merge(
        df,
        master_teams,
        on=["team", "season"],
        how="inner",
    )
    return df

def assign_player_positions(df):
    position_mapping = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    df['position'] = df['element_type'].map(position_mapping)
    return df

def load_gameweek_data(seasons):
    season_stats = pd.DataFrame()
    for season in seasons:
        for gw in range(1, 39):
            file_path = os.path.join(
                f"historical_data/Fantasy-Premier-League/data/{season}/gws", f"gw{gw}.csv"
            )
            try:
                gw_data = pd.read_csv(file_path, encoding="latin1").assign(
                    GW=gw, season=season
                )
                season_stats = pd.concat([season_stats, gw_data], ignore_index=True)
            except FileNotFoundError:
                logging.warning(f"File not found for Season {season} Gameweek {gw}")
    return season_stats

def balance_dataset(df):
    non_zero_points = df[df['total_points'] > 0]
    zero_points = df[df['total_points'] == 0].sample(n=len(non_zero_points), random_state=42).reset_index(drop=True)
    balanced_df = pd.concat([non_zero_points, zero_points], ignore_index=True)
    return balanced_df.sort_values(by=['name', 'season', 'GW'])

def feature_engineering(df, n_weeks):
    df["value"] = pd.to_numeric(df["value"]) / 10
    df["value"] = round(df["value"], 1)
    df['total_points'] = df['total_points'].fillna(0).astype(int)

    df['player_change'] = df['name'].ne(df['name'].shift()).cumsum()
    columns_to_shift = [
        "assists", "big_chances_created", "big_chances_missed", "bonus",
        "clean_sheets", "clearances_blocks_interceptions", "completed_passes",
        "dribbles", "fouls", "goals_conceded", "goals_scored",
        "key_passes", "minutes", "penalties_missed", "penalties_saved",
        "red_cards", "saves", "tackles", "total_points", "yellow_cards",
    ]
    for column in columns_to_shift:
        for lag in range(1, n_weeks + 1):
            df[f"previous_{column}_T{lag}"] = df.groupby('player_change')[column].shift(lag).fillna(0)
    
    df = df.drop('player_change', axis=1)
    df = pd.get_dummies(df, columns=['team_name', 'position'])
    drop_columns = [
        'assists', 'attempted_passes', 'big_chances_created', 'big_chances_missed',
        'bonus', 'bps', 'clean_sheets', 'clearances_blocks_interceptions',
        'completed_passes', 'dribbles', 'ea_index', 'element', 
        'errors_leading_to_goal', 'errors_leading_to_goal_attempt', 'fouls',
        'goals_conceded', 'goals_scored', 'id', 'key_passes', 'kickoff_time',
        'kickoff_time_formatted', 'loaned_in', 'loaned_out', 'minutes',
        'offside', 'open_play_crosses', 'own_goals', 'penalties_saved',
        'penalties_missed', 'penalties_conceded', 'recoveries', 'red_cards',
        'saves', 'selected', 'tackled', 'tackles', 'target_missed', 'team_a_score',
        'team_h_score', 'transfers_balance', 'transfers_in', 'transfers_out',
        'winning_goals', 'yellow_cards', 'team_x', 'xP', 'expected_assists',
        'expected_goal_involvements', 'expected_goals',
        'expected_goals_conceded', 'starts', 'team', 'element_type'
    ]
    df = df.drop(drop_columns, axis=1)
    return df

def prepare_features_targets(df, n_weeks):
    feature_cols = [
        'influence', 'ict_index', 'threat', 'creativity', 'value'
    ] + [f"previous_total_points_T{lag}" for lag in range(1, n_weeks + 1)]
    
    X = df[feature_cols].iloc[n_weeks:]
    y = df['total_points'].iloc[n_weeks:]
    return train_test_split(X, y, test_size=0.1, random_state=42)

def train_and_tune_model(X_train, y_train):
    param_distributions = {
        'learning_rate': [0.1],
        'n_estimators': [50],
        'max_depth': [5],
        'min_samples_split': [5]
    }
    
    model = GradientBoostingRegressor(random_state=42)
    
    randomized_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=50,
        scoring='neg_mean_absolute_error',
        cv=5,
        verbose=3,
        random_state=42,
        n_jobs=-1
    )
    
    logging.info("Starting hyperparameter tuning...")
    randomized_search.fit(X_train, y_train.values.ravel())
    logging.info("Hyperparameter tuning completed.")
    logging.info(f"Best parameters: {randomized_search.best_params_}")
    
    return randomized_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logging.info(f"Mean Absolute Error: {mae}")
    logging.info(f"Mean Squared Error: {mse}")
    logging.info(f"R-squared Score: {r2}")
    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("R-squared Score:", r2)

def main():
    seasons = [
        "2016-17", "2017-18", "2018-19", "2019-20",
        "2020-21", "2021-22", "2022-23", "2023-24"
    ]
    
    n_weeks = 1  # Customize the number of previous gameweeks
    
    logging.info("Creating code to name mapping...")
    code_to_name = create_code_to_name_mapping(seasons)
    
    logging.info("Loading gameweek data...")
    season_stats = load_gameweek_data(seasons)
    
    logging.info("Cleaning player names...")
    season_stats = clean_player_names(season_stats)
    
    logging.info("Merging with code to name mapping...")
    merged_df = pd.merge(
        season_stats,
        code_to_name[["code", "name", "season", "team", "element_type"]],
        on=["name", "season"],
        how="inner",
    ).rename(columns={'team_y': 'team'})
    
    logging.info("Merging team names and assigning positions...")
    merged_df = merge_team_names(merged_df)
    merged_df = assign_player_positions(merged_df)
    
    logging.info("Balancing the dataset...")
    balanced_df = balance_dataset(merged_df)
    
    logging.info("Performing feature engineering...")
    engineered_df = feature_engineering(balanced_df, n_weeks)
    
    logging.info("Preparing features and targets...")
    X_train, X_test, y_train, y_test = prepare_features_targets(engineered_df, n_weeks)
    
    logging.info("Starting model training and hyperparameter tuning...")
    best_model = train_and_tune_model(X_train, y_train)
    
    model_file = 'fpl_gbr_model.pkl'
    logging.info(f"Saving the best model to {model_file}...")
    joblib.dump(best_model, model_file)
    
    logging.info("Evaluating the model...")
    evaluate_model(best_model, X_test, y_test)
    
    logging.info("Model training and evaluation completed.")

if __name__ == "__main__":
    main()