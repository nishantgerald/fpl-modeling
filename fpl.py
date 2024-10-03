#!/usr/bin/env python3

import pandas as pd
import requests
import sys
import json

BASE_URL = 'https://fantasy.premierleague.com/api/'
API_URL = "http://localhost:5002/predict_points"

def fetch_json(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Request failed for {url}: {e}")
        sys.exit(1)

def load_players(data):
    players = pd.DataFrame(data['elements'])
    numeric_cols = ['ict_index', 'influence', 'creativity', 'threat', 'now_cost']
    players[numeric_cols] = players[numeric_cols].apply(pd.to_numeric, errors='coerce')
    players['photo'] = ("https://resources.premierleague.com/premierleague/photos/players/110x140/p" + players['photo']).str.replace('.jpg', '.png')
    players['now_cost'] = round(players['now_cost'] / 10, 1)
    players = players.sort_values(['second_name', 'first_name']).reset_index(drop=True).sort_values(by='id')
    players = players.sort_values(by='form')
    return players

def load_gameweeks(data):
    gw_info = pd.DataFrame(data['events']).sort_values(by='id')
    try:
        latest_gw_series = gw_info[(gw_info.is_current) & (gw_info.data_checked)].id
        latest_gw = int(latest_gw_series.iloc[0]) if not latest_gw_series.empty else None
    except:
        latest_gw_series = gw_info[(gw_info.is_previous) & (gw_info.data_checked)].id
        latest_gw = int(latest_gw_series.iloc[0]) if not latest_gw_series.empty else None
    
    if latest_gw is None:
        print("Unable to determine the latest gameweek.")
        sys.exit(1)
    
    return latest_gw

def load_fixtures():
    fixtures_data = fetch_json(BASE_URL + 'fixtures/')
    fixtures = pd.DataFrame(fixtures_data)[['event', 'team_a', 'team_h', 'team_a_score', 'team_h_score', 'kickoff_time', 'finished']]
    fixtures = fixtures.rename(columns={'event': 'GW'})
    return fixtures

def get_gw_data(gw):
    url = f"{BASE_URL}event/{gw}/live/"
    response = fetch_json(url)
    elements = response.get('elements', [])
    if not elements:
        print(f"No elements found for GW{gw}. Skipping this gameweek.")
        return pd.DataFrame()
    
    df = pd.json_normalize(elements, sep='_').drop(columns=[col for col in elements[0].keys() if col.startswith('explain')], errors='ignore')
    df = df.rename(columns=lambda x: x.replace('stats_', ''))
    df['GW'] = gw
    return df

def aggregate_gw_data(latest_gw):
    dfs = []
    for gw in range(1, latest_gw + 2):
        gw_df = get_gw_data(gw)
        if not gw_df.empty:
            dfs.append(gw_df)
    if not dfs:
        print("No gameweek data available to aggregate.")
        sys.exit(1)
    combined_df = pd.concat(dfs, ignore_index=True).sort_values(by=['id', 'GW'])
    return combined_df

def add_previous_stats(df):
    columns_to_shift = [
        "assists", "bonus", "clean_sheets", "creativity", "goals_conceded",
        "goals_scored", "ict_index", "influence", "minutes",
        "penalties_missed", "penalties_saved", "red_cards",
        "saves", "threat", "total_points", "yellow_cards",
    ]
    df['player_change'] = df['id'].ne(df['id'].shift()).cumsum()
    for column in columns_to_shift:
        df[f"previous_{column}"] = df.groupby('player_change')[column].shift(1).fillna(0)
    df = df.drop('player_change', axis=1)
    return df

def merge_player_data(df, players):
    return df.merge(
        players[['id', 'first_name', 'second_name', 'team', 'element_type', 'now_cost', 'code', 'photo']],
        on='id',
        how='left'
    ).rename(columns={
        'team': 'player_team_id',
        'element_type': 'position_id',
        'now_cost': 'player_value',
        'code': 'player_code',
        'photo': 'player_photo_url'
    })

def find_fixture(row, fixtures):
    fixture = fixtures[(fixtures['GW'] == row['GW']) & 
                      ((fixtures['team_a'] == row['player_team_id']) | 
                       (fixtures['team_h'] == row['player_team_id']))]
    if len(fixture) == 1:
        was_home = fixture['team_h'].values[0] == row['player_team_id']
        return pd.Series({
            'team_a': fixture['team_a'].values[0],
            'team_h': fixture['team_h'].values[0],
            'team_h_score': fixture['team_h_score'].values[0],
            'team_a_score': fixture['team_a_score'].values[0],
            'is_home_game': was_home
        })
    else:
        return pd.Series({'team_a': None, 'team_h': None, 'team_h_score': None, 'team_a_score': None, 'is_home_game': None})

def assign_fixture_data(df, fixtures):
    fixture_data = df.apply(lambda row: find_fixture(row, fixtures), axis=1)
    return pd.concat([df, fixture_data], axis=1)

def map_positions(df):
    position_mapping = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    df['player_position'] = df['position_id'].map(position_mapping)
    return df

def load_teams(data):
    teams = pd.DataFrame(data['teams'])[['id', 'name']]
    return teams

def assign_team_names(df, teams):
    def _assign(row):
        if row['is_home_game']:
            player_team_id = row['team_h']
            opposition_team_id = row['team_a']
        else:
            player_team_id = row['team_a']
            opposition_team_id = row['team_h']
        
        player_team_name = teams.loc[teams['id'] == player_team_id, 'name'].values[0] if player_team_id in teams['id'].values else None
        opposition_team_name = teams.loc[teams['id'] == opposition_team_id, 'name'].values[0] if opposition_team_id in teams['id'].values else None
        return pd.Series({'player_team_name': player_team_name, 'opposition_team_name': opposition_team_name})
    
    return df.apply(_assign, axis=1)

def calculate_forms(df):
    df['player_form'] = df.groupby('id')['total_points'].shift(1).fillna(0).add(
                        df.groupby('id')['total_points'].shift(2).fillna(0)
                      ).add(
                        df.groupby('id')['total_points'].shift(3).fillna(0)
                      ) / 3
    df['team_points'] = df.apply(determine_team_points, axis=1)
    df['team_form'] = df.groupby(['player_team_id'])['team_points'].shift(1).fillna(0).add(
                      df.groupby(['player_team_id'])['team_points'].shift(2).fillna(0)
                    ).add(
                      df.groupby(['player_team_id'])['team_points'].shift(3).fillna(0)
                    ) / 3
    return df

def determine_team_points(x):
    if (x['is_home_game'] and x['team_h_score'] > x['team_a_score']) or \
       (not x['is_home_game'] and x['team_a_score'] > x['team_h_score']):
        return 3
    elif x['team_h_score'] == x['team_a_score']:
        return 1
    else:
        return 0

def clean_dataframe(df):
    columns_to_drop = [
        'total_points', 'team_points', 'team_h_score', 'team_a_score',
        'previous_penalties_missed', 'previous_penalties_saved', 'previous_influence',
        'previous_creativity', 'previous_threat', 'minutes', 'goals_scored',
        'assists', 'clean_sheets', 'goals_conceded', 'own_goals',
        'penalties_saved', 'penalties_missed', 'yellow_cards',
        'red_cards', 'saves', 'bonus', 'bps', 'influence', 'creativity',
        'threat', 'ict_index', 'starts', 'expected_goals',
        'expected_assists', 'expected_goal_involvements',
        'expected_goals_conceded', 'in_dreamteam'
    ]
    return df.drop(columns=columns_to_drop, errors='ignore')

def filter_latest_gw(df, latest_gw):
    return df[df.GW == latest_gw]

def fill_na_values(df):
    df['player_form'] = df['player_form'].fillna(0)
    df['team_form'] = df['team_form'].fillna(0)
    return df

def one_hot_encode(df):
    df = pd.get_dummies(df, columns=['player_position', 'player_team_name', 'opposition_team_name'])
    df['is_home_game'] = df['is_home_game'].astype(int)
    return df

def ensure_required_features(df, required_features):
    missing_fields = required_features - set(df.columns)
    for field in missing_fields:
        df[field] = 0
    return df

def get_required_features():
    return {
        "player_value", "is_home_game", "previous_assists", "previous_bonus",
        "previous_clean_sheets", "previous_goals_conceded", "previous_goals_scored",
        "previous_ict_index", "previous_minutes", "previous_red_cards",
        "previous_saves", "previous_total_points", "previous_yellow_cards",
        "player_form", "team_form", "player_position_DEF", "player_position_FWD",
        "player_position_GKP", "player_position_MID",
        # Team names (ensure all team names are included)
        "player_team_name_Arsenal", "opposition_team_name_Arsenal",
        "player_team_name_Aston Villa", "opposition_team_name_Aston Villa",
        "player_team_name_Bournemouth", "opposition_team_name_Bournemouth",
        "player_team_name_Brentford", "opposition_team_name_Brentford",
        "player_team_name_Brighton", "opposition_team_name_Brighton",
        "player_team_name_Burnley", "opposition_team_name_Burnley",
        "player_team_name_Cardiff", "opposition_team_name_Cardiff",
        "player_team_name_Chelsea", "opposition_team_name_Chelsea",
        "player_team_name_Crystal Palace", "opposition_team_name_Crystal Palace",
        "player_team_name_Everton", "opposition_team_name_Everton",
        "player_team_name_Fulham", "opposition_team_name_Fulham",
        "player_team_name_Huddersfield", "opposition_team_name_Huddersfield",
        "player_team_name_Hull", "opposition_team_name_Hull",
        "player_team_name_Leeds", "opposition_team_name_Leeds",
        "player_team_name_Leicester", "opposition_team_name_Leicester",
        "player_team_name_Liverpool", "opposition_team_name_Liverpool",
        "player_team_name_Luton", "opposition_team_name_Luton",
        "player_team_name_Man City", "opposition_team_name_Man City",
        "player_team_name_Man Utd", "opposition_team_name_Man Utd",
        "player_team_name_Middlesbrough", "opposition_team_name_Middlesbrough",
        "player_team_name_Newcastle", "opposition_team_name_Newcastle",
        "player_team_name_Norwich", "opposition_team_name_Norwich",
        "player_team_name_Nott'm Forest", "opposition_team_name_Nott'm Forest",
        "player_team_name_Sheffield Utd", "opposition_team_name_Sheffield Utd",
        "player_team_name_Southampton", "opposition_team_name_Southampton",
        "player_team_name_Spurs", "opposition_team_name_Spurs",
        "player_team_name_Stoke", "opposition_team_name_Stoke",
        "player_team_name_Sunderland", "opposition_team_name_Sunderland",
        "player_team_name_Swansea", "opposition_team_name_Swansea",
        "player_team_name_Watford", "opposition_team_name_Watford",
        "player_team_name_West Brom", "opposition_team_name_West Brom",
        "player_team_name_West Ham", "opposition_team_name_West Ham",
        "player_team_name_Wolves", "opposition_team_name_Wolves",
    }

def prepare_features(df, required_features):
    df = one_hot_encode(df)
    df = ensure_required_features(df, required_features)
    return df

def load_and_prepare_data():
    data = fetch_json(BASE_URL + 'bootstrap-static/')
    players = load_players(data)
    latest_gw = load_gameweeks(data)
    fixtures = load_fixtures()
    gw_data = aggregate_gw_data(latest_gw)
    gw_data = add_previous_stats(gw_data)
    gw_data = merge_player_data(gw_data, players)
    gw_data = assign_fixture_data(gw_data, fixtures)
    gw_data = map_positions(gw_data)
    teams = load_teams(data)
    gw_data[['player_team_name', 'opposition_team_name']] = assign_team_names(gw_data, teams)
    gw_data = calculate_forms(gw_data)
    gw_data = clean_dataframe(gw_data)
    gw_data = filter_latest_gw(gw_data, latest_gw)
    gw_data = fill_na_values(gw_data)
    required_features = get_required_features()
    gw_data = prepare_features(gw_data, required_features)
    return gw_data, latest_gw

def get_prediction(row, required_features):
    input_data = {field: row.get(field, 0) for field in required_features}
    input_data['is_home_game'] = int(input_data.get('is_home_game', 0))
    try:
        response = requests.post(API_URL, json=input_data)
        response.raise_for_status()
        json_response = response.json()
        return json_response.get('predicted_points', None)
    except requests.exceptions.RequestException as e:
        print(f"Error for player {row['id']}: {e}")
        return None
    except json.JSONDecodeError:
        print(f"Invalid JSON response for player {row['id']}: {response.text}")
        return None

def add_predictions(df, required_features):
    df['player_predicted_points'] = df.apply(lambda row: get_prediction(row, required_features), axis=1)
    df['player_predicted_points'] = df['player_predicted_points'].fillna(-1)
    return df

def main():
    df, latest_gw = load_and_prepare_data()
    required_features = get_required_features()
    df = add_predictions(df, required_features)
    print("Predictions completed. Sample of results:")
    print(df.sort_values(by="player_predicted_points", ascending=False)[[
        'id', 'first_name', 'second_name', 'is_home_game', 'player_value',
        'player_form', 'team_form', 'player_predicted_points'
    ]].head(20))

if __name__ == "__main__":
    main()