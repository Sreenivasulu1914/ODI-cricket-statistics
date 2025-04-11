from flask import Flask, render_template, request
import pandas as pd
import requests
from sqlalchemy import create_engine
from flask_paginate import Pagination, get_page_args
from flask import jsonify
from flask_caching import Cache
import matplotlib.pyplot as plt
import seaborn as sns
import io
import numpy as np
import base64
import joblib
from sqlalchemy.pool import QueuePool
from groq import Groq
import os
import logging


app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})
# CORS(app)


# Set up logging globally
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# Load the model
# model = joblib.load('random_forest_model.pkl')

def get_db_connection():
    try:
        db_user = os.environ.get('DB_USER', 'root')
        db_pass = os.environ.get('DB_PASS', 'sGwKeLDmtoSojXHPByuZmLLRaKjXbJlw')  # Default for local testing
        db_host = os.environ.get('DB_HOST', 'yamanote.proxy.rlwy.net')
        db_port = os.environ.get('DB_PORT', '18381')
        db_name = os.environ.get('DB_NAME', 'railway')
        connection_string = f'mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}'
        logger.info(f"Connecting to: {connection_string}")
        return create_engine(connection_string, poolclass=QueuePool, pool_size=10, max_overflow=20)
    except Exception as e:
        logger.error(f"DB Connection failed: {str(e)}")
        return None

engine = get_db_connection()


# Set up Groq client with your API key
client = Groq(api_key="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")  # Replace with your Groq API key







@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    try:
        # Send message to Groq
        completion = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[
                {"role": "system", "content": "You are a helpful cricket assistant."},
                {"role": "user", "content": user_message}
            ],
            temperature=0.6,
            max_tokens=4096,
            top_p=0.95,
            stream=False
        )

        # Get the raw reply
        raw_reply = completion.choices[0].message.content

        # Clean up unwanted tags (e.g., <think>)
        import re
        clean_reply = re.sub(r'<think>.*?</think>', '', raw_reply, flags=re.DOTALL)  # Remove <think> tags
        clean_reply = clean_reply.strip()  # Remove extra whitespace

        return jsonify({'reply': clean_reply})
    except Exception as e:
        print("Groq Error:", str(e))
        return jsonify({'error': f"Chat failed: {str(e)}"}), 500











#------------------------------------------------------------------------------------------------



# Database connection with pooling
def get_db_connection():
    engine = create_engine('mysql+mysqlconnector://root:Seenu%40123@localhost/odi_cricket', 
                           poolclass=QueuePool, pool_size=10, max_overflow=20)
    return engine

engine = get_db_connection()

# # Load pre-trained models and data
# batsman_model = joblib.load('batsman_runs_model.pkl')
# batsman_df = joblib.load('batsman_data.pkl')
# batting_order_model = joblib.load('batting_order_model.pkl')
# players_df = joblib.load('recent_players.pkl')



# Route to fetch all player names for batsman runs dropdown
@app.route('/get_players', methods=['GET'])
def get_players():
    players = batsman_df['player_name'].unique().tolist()
    return jsonify(players)

# Route to fetch all unique teams for batting order dropdowns
@app.route('/get_teams', methods=['GET'])
def get_teams():
    teams = players_df['batting_team'].unique().tolist()
    return jsonify(teams)

# Route to predict runs for a selected batsman
@app.route('/predict', methods=['POST'])
def predict():
    batsman_name = request.json.get('batsman_name')
    if not batsman_name:
        return jsonify({'error': 'No batsman name provided'}), 400

    player_id = batsman_df[batsman_df['player_name'] == batsman_name]['player_id'].unique()
    if len(player_id) == 0:
        return jsonify({'error': 'Batsman not found'}), 404
    
    player_id = player_id[0]
    batsman_data = batsman_df[batsman_df['player_id'] == player_id].sort_values('Match_Date', ascending=False)
    if len(batsman_data) < 5:
        return jsonify({'error': 'Not enough data (less than 5 matches) for prediction'}), 400
    
    last_5_runs = batsman_data['runs'].head(5).values
    features = [[last_5_runs[0], last_5_runs[1], last_5_runs[2], last_5_runs[3], last_5_runs[4]]]
    predicted_runs = batsman_model.predict(features)[0]
    
    return jsonify({'batsman': batsman_name, 'predicted_runs': round(predicted_runs, 2)})

# Route to predict batting order
@app.route('/predict_batting_order', methods=['POST'])
def predict_batting_order():
    data = request.json
    batting_team = data.get('batting_team')
    venue = data.get('venue', 'default_stadium')
    opposition = data.get('opposition', 'default_team')
    batting_first = int(data.get('batting_first', 1))  # 1 for batting first, 0 for chasing

    # Fetch avg_stadium_runs for the venue
    venue_query = f"SELECT AVG(Team1_Runs_Scored + Team2_Runs_Scored) as avg_stadium_runs FROM matches WHERE Match_Venue_Stadium = '{venue}'"
    avg_stadium_runs = pd.read_sql(venue_query, engine)['avg_stadium_runs'].iloc[0] or 300  # Default to 300 if NaN

    # Select top 11 recent players for the batting team
    selected_players = players_df[players_df['batting_team'] == batting_team]['player_id'].head(11).tolist()
    if len(selected_players) < 11:
        return jsonify({'error': f'Not enough players for {batting_team}'}), 400

    # Predict runs for each player in each position
    predictions = {}
    feature_cols = batting_order_model.feature_names_in_
    for player_id in selected_players:
        player_preds = []
        for pos in range(1, 12):
            features = pd.DataFrame([[pos, batting_first, avg_stadium_runs] + [0] * (len(feature_cols) - 3)], columns=feature_cols)
            if f'Match_Venue_Stadium_{venue}' in feature_cols:
                features[f'Match_Venue_Stadium_{venue}'] = 1
            if f'opposition_{opposition}' in feature_cols:
                features[f'opposition_{opposition}'] = 1
            pred_runs = batting_order_model.predict(features)[0]
            player_preds.append((pos, pred_runs))
        predictions[player_id] = player_preds

    # Optimize batting order (greedy approach)
    batting_order = []
    used_positions = set()
    for _ in range(11):
        best_player = None
        best_pos = None
        best_runs = -1
        for player_id, preds in predictions.items():
            if player_id in [p[0] for p in batting_order]:
                continue
            for pos, runs in preds:
                if pos not in used_positions and runs > best_runs:
                    best_player = player_id
                    best_pos = pos
                    best_runs = runs
        batting_order.append((best_player, best_pos, best_runs))
        used_positions.add(best_pos)

    # Format response
    result = [{'player_name': players_df[players_df['player_id'] == pid]['player_name'].iloc[0], 
               'position': pos, 'predicted_runs': round(runs, 2)} 
              for pid, pos, runs in batting_order]
    return jsonify({'batting_order': result})

#----------------------------------------------------------------------

def get_player_details(player_name):
    """Fetch player details from the database."""
    conn = get_db_connection()
    query = "SELECT * FROM players WHERE LOWER(player_name) = LOWER(%s)"
    player_data = pd.read_sql(query, conn, params=(player_name,))
    conn.dispose()
    return player_data.to_dict(orient='records')

def get_match_details(match_id):
    """Fetch match details from the database."""
    conn = get_db_connection()
    query = "SELECT * FROM matches WHERE Match_ID = %s"
    match_data = pd.read_sql(query, conn, params=(match_id,))
    conn.dispose()
    return match_data.to_dict(orient='records')

def get_team_details(team_name):
    """Fetch team details from the database."""
    conn = get_db_connection()
    query = """
    SELECT p.player_name, p.dob, p.batting_style, p.bowling_style, p.image_url
    FROM players p
    JOIN countries c ON p.country_id = c.country_id
    WHERE LOWER(c.country_name) = LOWER(%s)
    """
    team_data = pd.read_sql(query, conn, params=(team_name,))
    conn.dispose()
    return team_data.to_dict(orient='records')



@app.route('/get_player_details', methods=['GET'])
def player_details():
    """API endpoint to fetch player details."""
    player_name = request.args.get('player_name')
    details = get_player_details(player_name)
    return jsonify(details)

@app.route('/get_match_details', methods=['GET'])
def match_details():
    """API endpoint to fetch match details."""
    match_id = request.args.get('match_id')
    details = get_match_details(match_id)
    return jsonify(details)

@app.route('/get_team_details', methods=['GET'])
def team_details():
    """API endpoint to fetch team details."""
    team_name = request.args.get('team_name')
    details = get_team_details(team_name)
    return jsonify(details)

@app.route('/current_matches', methods=['GET'])
def current_matches():
    api_key = "975a7e1f-2ab2-4a0b-bddf-e22c630aa9df"
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 10, type=int)
    offset = (page - 1) * limit
    url = f"https://api.cricapi.com/v1/currentMatches?apikey={api_key}&offset={offset}&limit={limit}"

    response = requests.get(url)
    if response.status_code == 200:
        matches_data = response.json()
        matches = matches_data.get('data', [])
        
        # If the total count is not available, assume pagination based on returned results
        total_matches = matches_data.get('info', {}).get('total', len(matches))

        return jsonify({'matches': matches, 'total': total_matches})
    else:
        return jsonify({'error': 'Failed to fetch current matches'}), response.status_code

@app.route('/search_players', methods=['GET'])
def search_players():
    query = request.args.get('query')
    api_key = "975a7e1f-2ab2-4a0b-bddf-e22c630aa9df"
    url = f"https://api.cricapi.com/v1/players?apikey={api_key}&offset=0&search={query}"
    response = requests.get(url)
    if response.status_code == 200:
        players = response.json().get('data', [])
        return jsonify(players)
    else:
        return jsonify({'error': 'Failed to fetch player data'}), response.status_code

@app.route('/player_detail/<player_id>', methods=['GET'])
def player_detail(player_id):
    api_key = "975a7e1f-2ab2-4a0b-bddf-e22c630aa9df"
    url = f"https://api.cricapi.com/v1/players_info?apikey={api_key}&id={player_id}"
    response = requests.get(url)
    if response.status_code == 200:
        player_info = response.json().get('data', {})
        return render_template('player_details.html', player=player_info)
    else:
        return jsonify({'error': 'Failed to fetch player info'}), response.status_code


@app.route('/match_detail/<match_id>', methods=['GET'])
def match_detail(match_id):
    api_key = "975a7e1f-2ab2-4a0b-bddf-e22c630aa9df"
    url = f"https://api.cricapi.com/v1/match_info?apikey={api_key}&id={match_id}"
    response = requests.get(url)
    if response.status_code == 200:
        match_info = response.json().get('data', {})
        print(match_info)
        return render_template('match_details.html', match=match_info)
    else:
        return jsonify({'error': 'Failed to fetch match info'}), response.status_code
    

@app.route('/ex')
def ex():
    return render_template('ex.html')


def fetch_data():
    conn = get_db_connection()
    batting_data = pd.read_sql("SELECT * FROM batting_performances", conn)
    bowling_data = pd.read_sql("SELECT * FROM bowling_performances", conn)
    players_data = pd.read_sql("SELECT * FROM players", conn)
    matches_data = pd.read_sql("SELECT * FROM matches", conn)
    conn.dispose()
    return batting_data, bowling_data, players_data, matches_data

def fetch_recent_matches():
    conn = get_db_connection()
    query = """
        SELECT 
            m.Match_ID as id, 
            m.Team1_Name as team1, 
            m.Team2_Name as team2, 
            m.Match_Date as date, 
            m.Match_Venue_Stadium as venue, 
            m.Match_Result_Text as result,
            c1.country_image as team1_image,
            c2.country_image as team2_image,
            m.Team1_Runs_Scored as team1_runs,
            m.Team1_Wickets_Fell as team1_wickets,
            m.Team2_Runs_Scored as team2_runs,
            m.Team2_Wickets_Fell as team2_wickets
        FROM matches m
        JOIN countries c1 ON m.Team1_ID = c1.country_id
        JOIN countries c2 ON m.Team2_ID = c2.country_id
        ORDER BY m.Match_Date DESC 
        LIMIT 8
    """
    recent_matches = pd.read_sql(query, conn)
    conn.dispose()
    return recent_matches.to_dict(orient='records')

@app.route('/')
@cache.cached(timeout=3600)
def home():
    # Fetch data
    batting_data, bowling_data, players_data, matches_data = fetch_data()
    
    recent_matches = fetch_recent_matches()
    print("---------------------------------------------------------------------------------------------------")
    print(recent_matches)
    print("-------------------------------------------------------------------------------------------------------------")
    
    conn = get_db_connection()
    
    years = pd.read_sql("SELECT DISTINCT YEAR(Match_Date) as year FROM matches ORDER BY year", conn)['year'].dropna().astype(int).tolist()
    teams = pd.read_sql("SELECT DISTINCT Team1_Name as team FROM matches UNION SELECT DISTINCT Team2_Name as team FROM matches ORDER BY team", conn)['team'].dropna().tolist()
    stadiums = pd.read_sql("SELECT DISTINCT Match_Venue_Stadium as stadium FROM matches ORDER BY stadium", conn)['stadium'].dropna().tolist()
    series = pd.read_sql("SELECT DISTINCT Series_Name as series FROM matches ORDER BY series", conn)['series'].dropna().tolist()
    
    matches_played = pd.read_sql("SELECT COUNT(*) as count FROM matches", conn)['count'].iloc[0]
    trophies_won = pd.read_sql("SELECT COUNT(DISTINCT Series_Name) as count FROM matches WHERE Match_Result_Text LIKE '%won%'", conn)['count'].iloc[0]
    players_count = pd.read_sql("SELECT COUNT(*) as count FROM players", conn)['count'].iloc[0]
    years_count = len(years)
    
    year = request.args.get('year')
    team = request.args.get('team')
    stadium = request.args.get('stadium')
    series_name = request.args.get('series')
    per_page = int(request.args.get('per_page', 10))
    page, _, offset = get_page_args(page_parameter='page', per_page_parameter='per_page')
    
    query = "SELECT Match_ID as id,Series_Name as series_name, Match_Name as match_name, Team1_Name as team1, Team2_Name as team2, Match_Winner as match_winner, Match_Result_Text as match_result, Match_Venue_Stadium as venue FROM matches WHERE 1=1"
    
    if year:
        query += f" AND YEAR(Match_Date) = {year}"
    if team:
        query += f" AND (Team1_Name = '{team}' OR Team2_Name = '{team}')"
    if stadium:
        query += f" AND Match_Venue_Stadium = '{stadium}'"
    if series_name:
        query += f" AND Series_Name = '{series_name}'"
    
    all_matches_df = pd.read_sql(query, conn)
    conn.dispose()
    
    total = len(all_matches_df)
    all_matches = all_matches_df.iloc[offset: offset + per_page].to_dict(orient='records')
    
    pagination = Pagination(page=page, per_page=per_page, total=total, css_framework='bootstrap4')

    return render_template('home.html', 
                           recent_matches=recent_matches, all_matches=all_matches, pagination=pagination, per_page=per_page, years=years, teams=teams, stadiums=stadiums, series=series, matches_played=matches_played, trophies_won=trophies_won, players_count=players_count, years_count=years_count
                        #    ,
                        #    plot_url_batting_order=plot_url_batting_order
                           )



@app.route('/get_counts')
@cache.cached(timeout=3600)
def get_counts():
    # Fetch data
    batting_data, bowling_data, players_data, matches_data = fetch_data()
    
    # Calculate counts
    num_players = players_data['player_id'].nunique()
    num_matches = matches_data['Match_ID'].nunique()
    num_years = matches_data['Match_Date'].dt.year.nunique()
    num_trophies = matches_data['Series_Name'].nunique()
    
    return jsonify({
        'num_players': num_players,
        'num_matches': num_matches,
        'num_years': num_years,
        'num_trophies': num_trophies
    })

@app.route('/players')
@cache.cached(timeout=3600)
def players():
    conn = get_db_connection()
    
    top_batsmen = pd.read_sql("""
    SELECT players.player_id, player_name, SUM(runs) as total_runs, image_url, country_name, country_image, players.country_id
    FROM batting_performances
    JOIN players ON batting_performances.batsman = players.player_id
    JOIN countries ON players.country_id = countries.country_id
    GROUP BY players.player_id, player_name, image_url, country_name, country_image, players.country_id
    ORDER BY total_runs DESC
    LIMIT 10
    """, conn).to_dict(orient='records')
    
    top_bowlers = pd.read_sql("""
        SELECT players.player_id, player_name, SUM(wickets) as total_wickets, image_url, country_name, country_image, players.country_id
        FROM bowling_performances
        JOIN players ON bowling_performances.bowler_id = players.player_id
        JOIN countries ON players.country_id = countries.country_id
        GROUP BY players.player_id, player_name, image_url, country_name, country_image, players.country_id
        ORDER BY total_wickets DESC
        LIMIT 10
    """, conn).to_dict(orient='records')
    
    top_batting_averages = pd.read_sql("""
        SELECT players.player_id, player_name, AVG(runs) as batting_average, image_url, country_name, country_image, players.country_id
        FROM batting_performances
        JOIN players ON batting_performances.batsman = players.player_id
        JOIN countries ON players.country_id = countries.country_id
        GROUP BY players.player_id, player_name, image_url, country_name, country_image, players.country_id
        ORDER BY batting_average DESC
        LIMIT 10
    """, conn).to_dict(orient='records')
    
    top_economy_bowlers = pd.read_sql("""
        SELECT players.player_id, player_name, AVG(economy) as economy_rate, image_url, country_name, country_image, players.country_id
        FROM bowling_performances
        JOIN players ON bowling_performances.bowler_id = players.player_id
        JOIN countries ON players.country_id = countries.country_id
        GROUP BY players.player_id, player_name, image_url, country_name, country_image, players.country_id
        ORDER BY economy_rate ASC
        LIMIT 10
    """, conn).to_dict(orient='records')
    
    conn.dispose()
    
    return render_template('players.html', top_batsmen=top_batsmen, top_bowlers=top_bowlers, top_batting_averages=top_batting_averages, top_economy_bowlers=top_economy_bowlers)

@app.route('/load_more_batsmen')
def load_more_batsmen():
    offset = request.args.get('offset', 0, type=int)
    conn = get_db_connection()
    more_batsmen = pd.read_sql(f"""
        SELECT players.player_id, player_name, SUM(runs) as total_runs, image_url, country_name, country_image, players.country_id
        FROM batting_performances
        JOIN players ON batting_performances.batsman = players.player_id
        JOIN countries ON players.country_id = countries.country_id
        GROUP BY players.player_id, player_name, image_url, country_name, country_image, players.country_id
        ORDER BY total_runs DESC
        LIMIT 10 OFFSET {offset}
    """, conn).to_dict(orient='records')
    conn.dispose()
    return jsonify(more_batsmen)

@app.route('/load_more_bowlers')
def load_more_bowlers():
    offset = request.args.get('offset', 0, type=int)
    conn = get_db_connection()
    more_bowlers = pd.read_sql(f"""
        SELECT players.player_id, player_name, SUM(wickets) as total_wickets, image_url, country_name, country_image, players.country_id
        FROM bowling_performances
        JOIN players ON bowling_performances.bowler_id = players.player_id
        JOIN countries ON players.country_id = countries.country_id
        GROUP BY players.player_id, player_name, image_url, country_name, country_image, players.country_id
        ORDER BY total_wickets DESC
        LIMIT 10 OFFSET {offset}
    """, conn).to_dict(orient='records')
    conn.dispose()
    return jsonify(more_bowlers)

@app.route('/load_more_batting_averages')
def load_more_batting_averages():
    offset = request.args.get('offset', 0, type=int)
    conn = get_db_connection()
    more_batting_averages = pd.read_sql(f"""
        SELECT players.player_id, player_name, AVG(runs) as batting_average, image_url, country_name, country_image, players.country_id
        FROM batting_performances
        JOIN players ON batting_performances.batsman = players.player_id
        JOIN countries ON players.country_id = countries.country_id
        GROUP BY players.player_id, player_name, image_url, country_name, country_image, players.country_id
        ORDER BY batting_average DESC
        LIMIT 10 OFFSET {offset}
    """, conn).to_dict(orient='records')
    conn.dispose()
    return jsonify(more_batting_averages)

@app.route('/load_more_economy_bowlers')
def load_more_economy_bowlers():
    offset = request.args.get('offset', 0, type=int)
    conn = get_db_connection()
    more_economy_bowlers = pd.read_sql(f"""
        SELECT players.player_id, player_name, AVG(economy) as economy_rate, image_url, country_name, country_image, players.country_id
        FROM bowling_performances
        JOIN players ON bowling_performances.bowler_id = players.player_id
        JOIN countries ON players.country_id = countries.country_id
        GROUP BY players.player_id, player_name, image_url, country_name, country_image, players.country_id
        ORDER BY economy_rate ASC
        LIMIT 10 OFFSET {offset}
    """, conn).to_dict(orient='records')
    conn.dispose()
    return jsonify(more_economy_bowlers)





@app.route('/player_info/<int:player_id>')
@cache.cached(timeout=3600)
def player_info(player_id):
    conn = get_db_connection()
    
    player_details = pd.read_sql("""
        SELECT player_name, image_url, dob, country_name, country_image, batting_style, bowling_style
        FROM players
        JOIN countries ON players.country_id = countries.country_id
        WHERE player_id = %s
    """, conn, params=(player_id,)).to_dict(orient='records')[0]
    
    batting_ranking = pd.read_sql("""
        SELECT player_id, player_name, RANK() OVER (ORDER BY SUM(runs) DESC) as ranking
        FROM batting_performances
        JOIN players ON batting_performances.batsman = players.player_id
        GROUP BY player_id, player_name
    """, conn).to_dict(orient='records')
    
    player_batting_ranking = next((item for item in batting_ranking if item["player_id"] == player_id), None)
    
    bowling_ranking = pd.read_sql("""
        SELECT player_id, player_name, RANK() OVER (ORDER BY SUM(wickets) DESC) as ranking
        FROM bowling_performances
        JOIN players ON bowling_performances.bowler_id = players.player_id
        GROUP BY player_id, player_name
    """, conn).to_dict(orient='records')
    
    player_bowling_ranking = next((item for item in bowling_ranking if item["player_id"] == player_id), None)
    
    strike_rate_data = []
    if player_batting_ranking:
        strike_rate_data = pd.read_sql("""
            SELECT YEAR(Match_Date) as year, AVG(strike_rate) as avg_strike_rate
            FROM batting_performances
            JOIN matches ON batting_performances.match_id = matches.Match_ID
            WHERE batsman = %s
            GROUP BY YEAR(Match_Date)
            ORDER BY year
        """, conn, params=(player_id,)).to_dict(orient='records')
    
    economy_rate_data = []
    if player_bowling_ranking:
        economy_rate_data = pd.read_sql("""
            SELECT YEAR(Match_Date) as year, AVG(economy) as avg_economy_rate
            FROM bowling_performances
            JOIN matches ON bowling_performances.match_id = matches.Match_ID
            WHERE bowler_id = %s
            GROUP BY YEAR(Match_Date)
            ORDER BY year
        """, conn, params=(player_id,)).to_dict(orient='records')
    
    # Fetch runs vs strike rate data for scatter plot
    runs_vs_strike_rate = pd.read_sql("""
        SELECT runs, strike_rate
        FROM batting_performances
        WHERE batsman = %s
    """, conn, params=(player_id,)).to_dict(orient='records')
    
    # Fetch home vs away performance data
    home_away_performance = pd.read_sql("""
        SELECT 
            CASE 
                WHEN m.Match_Venue_Country = c.country_name THEN 'Home' 
                ELSE 'Away' 
            END as match_location,
            AVG(bp.runs) as avg_runs,
            AVG(bp.strike_rate) as avg_strike_rate
        FROM batting_performances bp
        JOIN matches m ON bp.match_id = m.Match_ID
        JOIN players p ON bp.batsman = p.player_id
        JOIN countries c ON p.country_id = c.country_id
        WHERE bp.batsman = %s
        GROUP BY match_location
    """, conn, params=(player_id,)).to_dict(orient='records')
    
    conn.dispose()



    df = pd.DataFrame(runs_vs_strike_rate)
    plt.figure(figsize=(10, 6))
    scatter_plot = sns.scatterplot(x='strike_rate', y='runs', data=df)
    scatter_plot.set_title('Runs vs Strike Rate')
    scatter_plot.set_xlabel('Strike Rate')
    scatter_plot.set_ylabel('Runs')
    
    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    
    # Encode the image to base64
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    
    return render_template('player_info.html', player_details=player_details, batting_ranking=player_batting_ranking, bowling_ranking=player_bowling_ranking, strike_rate_data=strike_rate_data, economy_rate_data=economy_rate_data, runs_vs_strike_rate=runs_vs_strike_rate, home_away_performance=home_away_performance, plot_url=plot_url)



@app.route('/trends')
@cache.cached(timeout=3600)
def trends():
    conn = get_db_connection()
    
    # Fetch data for the last 5 years
    query = """
        SELECT bp.bowler_id, p.player_name, SUM(bp.wickets) as total_wickets
        FROM bowling_performances bp
        JOIN players p ON bp.bowler_id = p.player_id
        JOIN matches m ON bp.match_id = m.Match_ID
        WHERE YEAR(m.Match_Date) >= YEAR(CURDATE()) - 5
        GROUP BY bp.bowler_id, p.player_name
        ORDER BY total_wickets DESC
        LIMIT 10
    """
    consistent_wicket_takers = pd.read_sql(query, conn).to_dict(orient='records')

    query_winning_percentages = """
        SELECT Team1_Name as team_name, 
               SUM(CASE WHEN Match_Winner = Team1_Name THEN 1 ELSE 0 END) as wins,
               COUNT(*) as total_matches,
               (SUM(CASE WHEN Match_Winner = Team1_Name THEN 1 ELSE 0 END) / COUNT(*)) * 100 as winning_percentage
        FROM matches
        GROUP BY Team1_Name
        HAVING total_matches > 0
        ORDER BY winning_percentage DESC
    """
    winning_percentages = pd.read_sql(query_winning_percentages, conn).to_dict(orient='records')
    winning_percentages = winning_percentages[2:]

    query_toss_impact = """
        SELECT 
            Toss_Winner_Choice as toss_decision,
            SUM(CASE WHEN Match_Winner = Toss_Winner THEN 1 ELSE 0 END) as wins,
            COUNT(*) as total_matches,
            (SUM(CASE WHEN Match_Winner = Toss_Winner THEN 1 ELSE 0 END) / COUNT(*)) * 100 as win_percentage
        FROM matches
        WHERE Toss_Winner IS NOT NULL
        GROUP BY Toss_Winner_Choice
    """
    toss_impact = pd.read_sql(query_toss_impact, conn).to_dict(orient='records')

    query_chasing_300 = """
        SELECT 
            SUM(CASE WHEN Team2_Runs_Scored >= 300 AND Match_Winner = Team2_Name THEN 1 ELSE 0 END) as successful_chases,
            SUM(CASE WHEN Team2_Runs_Scored >= 300 THEN 1 ELSE 0 END) as total_chases
        FROM matches
    """
    chasing_300 = pd.read_sql(query_chasing_300, conn).to_dict(orient='records')[0]
    
    query_team_chasing_300 = """
        SELECT 
            Team2_Name as team_name,
            SUM(CASE WHEN Team2_Runs_Scored >= 300 AND Match_Winner = Team2_Name THEN 1 ELSE 0 END) as successful_chases,
            SUM(CASE WHEN Team2_Runs_Scored >= 300 THEN 1 ELSE 0 END) as total_chases,
            (SUM(CASE WHEN Team2_Runs_Scored >= 300 AND Match_Winner = Team2_Name THEN 1 ELSE 0 END) / SUM(CASE WHEN Team2_Runs_Scored >= 300 THEN 1 ELSE 0 END)) * 100 as success_percentage
        FROM matches
        GROUP BY Team2_Name
        HAVING total_chases > 0
        ORDER BY success_percentage DESC
    """
    team_chasing_300 = pd.read_sql(query_team_chasing_300, conn).to_dict(orient='records')
    team_chasing_300 = team_chasing_300[2:14]
    
    query_stadiums = """
        SELECT 
            Match_Venue_Stadium,
            COALESCE(AVG((Team1_Runs_Scored + Team2_Runs_Scored) / (Team1_Wickets_Fell + Team2_Wickets_Fell + 2)), 0) as Average_Run_Rate
        FROM matches
        GROUP BY Match_Venue_Stadium
        ORDER BY Average_Run_Rate DESC
    """
    all_stadiums_data = pd.read_sql(query_stadiums, conn)
    
    # Generate the horizontal bar chart using Matplotlib or Seaborn
    plt.figure(figsize=(10, 50))  # Increase the figure size
    sns.barplot(y='Match_Venue_Stadium', x='Average_Run_Rate', data=all_stadiums_data, palette="viridis")
    plt.ylabel("Stadium")
    plt.xlabel("Average Run Rate")
    plt.title("Variation of Run Rates Across Different Stadiums")
    plt.xticks(fontsize=20)  # Adjust font size
    
    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')  # Use bbox_inches to ensure labels are not cut off
    img.seek(0)
    
    # Encode the image as a base64 string
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
     

    conn.dispose()

    return render_template(
        'trends.html', 
        consistent_wicket_takers=consistent_wicket_takers, 
        winning_percentages=winning_percentages, 
        toss_impact=toss_impact, 
        chasing_300=chasing_300, 
        team_chasing_300=team_chasing_300, 
        plot_url=plot_url, 
        
    )

@app.route('/teams')
def teams():
    conn = get_db_connection()
    query = "SELECT country_name,country_id, COALESCE(country_image, 'default_image.png') as country_image FROM countries"
    teams = pd.read_sql(query, conn).to_dict(orient='records')
    
    print("TEAM NAMES AND IMAGES ")
    print("====================================================================================")
    print(teams)
    print("====================================================================================")
    conn.dispose()
    
    return render_template('teams.html', teams=teams)

@app.route('/team/<int:team_id>')
def team_players(team_id):
    conn = get_db_connection()
    query = """
        SELECT DISTINCT p.player_id, p.player_name, p.image_url
        FROM players p
        JOIN batting_performances bp ON p.player_id = bp.batsman
        JOIN matches m ON bp.match_id = m.Match_ID
        WHERE p.country_id = %s
        AND YEAR(m.Match_Date) = 2023
        UNION
        SELECT DISTINCT p.player_id, p.player_name, p.image_url
        FROM players p
        JOIN bowling_performances bw ON p.player_id = bw.bowler_id
        JOIN matches m ON bw.match_id = m.Match_ID
        WHERE p.country_id = %s
        AND YEAR(m.Match_Date) = 2023
    """
    players = pd.read_sql(query, conn, params=(team_id, team_id)).to_dict(orient='records')
    conn.dispose()
    return render_template('team_players.html', players=players)

@app.route('/match_analysis/<int:match_id>')
@cache.cached(timeout=3600)
def match_analysis(match_id):
    conn = get_db_connection()
    
    # Fetch match details
    match_query = """
        SELECT 
            m.Match_ID as id, 
            m.Match_Name as match_name,
            m.Series_Name as series_name,
            m.Match_Date as date,
            m.Toss_Winner as toss_winner,
            m.Match_Venue_Stadium as venue_stadium,
            m.Match_Venue_Country as venue_country,
            m.Umpire_1 as umpire_1,
            m.Umpire_2 as umpire_2,
            m.Match_Referee as match_referee,
            m.Team1_Name as team1, 
            m.Team2_Name as team2, 
            m.Team1_Runs_Scored as team1_runs, 
            m.Team2_Runs_Scored as team2_runs, 
            m.Team1_Wickets_Fell as team1_wickets,
            m.Team2_Wickets_Fell as team2_wickets,
            m.Team1_Extras_Rec as team1_extras,
            m.Team2_Extras_Rec as team2_extras,
            m.Match_Winner as winner,
            m.Team1_Playing_11 as team1_playing_11,
            m.Team2_Playing_11 as team2_playing_11,
            (SELECT AVG((runs / balls) * 100) 
             FROM batting_performances 
             WHERE match_id = m.Match_ID AND team = m.Team1_Name) as team1_strike_rate,
            (SELECT AVG((runs / balls) * 100) 
             FROM batting_performances 
             WHERE match_id = m.Match_ID AND team = m.Team2_Name) as team2_strike_rate
        FROM matches m
        WHERE m.Match_ID = %s
    """
    match = pd.read_sql(match_query, conn, params=(match_id,)).to_dict(orient='records')[0]
    
    # Convert team1_playing_11 and team2_playing_11 to a format suitable for SQL IN clause
    team1_ids = match['team1_playing_11'].strip('[]').replace("'", "").split(',')
    team2_ids = match['team2_playing_11'].strip('[]').replace("'", "").split(',')
    
    # Fetch player names for team1_playing_11
    team1_players_query = """
        SELECT player_name
        FROM players
        WHERE player_id IN (%s)
    """ % ','.join(team1_ids)
    team1_players = pd.read_sql(team1_players_query, conn)['player_name'].tolist()
    
    # Fetch player names for team2_playing_11
    team2_players_query = """
        SELECT player_name
        FROM players
        WHERE player_id IN (%s)
    """ % ','.join(team2_ids)
    team2_players = pd.read_sql(team2_players_query, conn)['player_name'].tolist()
    
    # Fetch batting performances for team1
    team1_batting_query = """
        SELECT p.player_name, bp.wicket_type, bp.bowler, bp.fielders, bp.runs, bp.balls, bp.fours, bp.sixes, bp.strike_rate
        FROM batting_performances bp
        JOIN players p ON bp.batsman = p.player_id
        WHERE bp.match_id = %s AND bp.team = %s
    """
    team1_batting = pd.read_sql(team1_batting_query, conn, params=(match_id, match['team1'])).to_dict(orient='records')
    
    # Fetch batting performances for team2
    team2_batting_query = """
        SELECT p.player_name, bp.wicket_type, bp.bowler, bp.fielders, bp.runs, bp.balls, bp.fours, bp.sixes, bp.strike_rate
        FROM batting_performances bp
        JOIN players p ON bp.batsman = p.player_id
        WHERE bp.match_id = %s AND bp.team = %s
    """
    team2_batting = pd.read_sql(team2_batting_query, conn, params=(match_id, match['team2'])).to_dict(orient='records')
    
    # Fetch fall of wickets for team1
    team1_fow_query = """
        SELECT overs, runs
        FROM fall_of_wickets
        WHERE match_id = %s AND team = %s
    """
    team1_fow = pd.read_sql(team1_fow_query, conn, params=(match_id, match['team1'])).to_dict(orient='records')
    
    # Fetch fall of wickets for team2
    team2_fow_query = """
        SELECT overs, runs
        FROM fall_of_wickets
        WHERE match_id = %s AND team = %s
    """
    team2_fow = pd.read_sql(team2_fow_query, conn, params=(match_id, match['team2'])).to_dict(orient='records')
    
    # Fetch bowling performances for team2 (bowling against team1)
    team2_bowling_query = """
        SELECT p.player_name, bp.overs, bp.maidens, bp.conceded as runs, bp.wickets, bp.economy, bp.dots, bp.fours, bp.sixes, bp.wides, bp.no_balls
        FROM bowling_performances bp
        JOIN players p ON bp.bowler_id = p.player_id
        WHERE bp.match_id = %s AND bp.team = %s
    """
    team2_bowling = pd.read_sql(team2_bowling_query, conn, params=(match_id, match['team2'])).to_dict(orient='records')
    
    # Fetch bowling performances for team1 (bowling against team2)
    team1_bowling_query = """
        SELECT p.player_name, bp.overs, bp.maidens, bp.conceded as runs, bp.wickets, bp.economy, bp.dots, bp.fours, bp.sixes, bp.wides, bp.no_balls
        FROM bowling_performances bp
        JOIN players p ON bp.bowler_id = p.player_id
        WHERE bp.match_id = %s AND bp.team = %s
    """
    team1_bowling = pd.read_sql(team1_bowling_query, conn, params=(match_id, match['team1'])).to_dict(orient='records')
    
    conn.dispose()
    
    
    return render_template('match_analysis.html', match=match, team1_players=team1_players, team2_players=team2_players,
                           team1_batting=team1_batting, team2_batting=team2_batting, team1_fow=team1_fow, team2_fow=team2_fow,
                           team2_bowling=team2_bowling, team1_bowling=team1_bowling)

# if __name__ == '__main__':
#     app.run(debug=True)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)