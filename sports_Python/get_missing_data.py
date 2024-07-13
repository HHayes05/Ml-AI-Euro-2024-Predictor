import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
from tabulate import tabulate

# Load the data
historical_results = pd.read_csv('cleaned_euro_results.csv')
fixtures = pd.read_csv('cleaned_euro_fixture.csv')
group_tables = pickle.load(open('dict_table','rb'))


# Prepare the data for machine learning
def prepare_data(df):
    X = df[['Home Team', 'Away Team']]
    y = np.where(df['Home Goals'] > df['Away Goals'], 'Home Win',
                 np.where(df['Home Goals'] < df['Away Goals'], 'Away Win', 'Draw'))
    return X, y

def prepare_goal_data(df):
    X = df[['Home Team', 'Away Team']]
    y_home = df['Home Goals']
    y_away = df['Away Goals']
    return X, y_home, y_away

X, y = prepare_data(historical_results)
X_goals, y_home_goals, y_away_goals = prepare_goal_data(historical_results)
# Get all unique teams
all_teams = set(X['Home Team']) | set(X['Away Team'])
for group in group_tables.values():
    all_teams |= set(group['Team'])

# Encode team names
le = LabelEncoder()
le.fit(list(all_teams))

# Create explicit copies
X = X[['Home Team', 'Away Team']].copy()
X_goals = X_goals[['Home Team', 'Away Team']].copy()

# Replace 'Germany (H)' with 'Germany'
X.loc[:, 'Home Team'] = X['Home Team'].replace('Germany (H)', 'Germany')
X.loc[:, 'Away Team'] = X['Away Team'].replace('Germany (H)', 'Germany')
X_goals.loc[:, 'Home Team'] = X_goals['Home Team'].replace('Germany (H)', 'Germany')
X_goals.loc[:, 'Away Team'] = X_goals['Away Team'].replace('Germany (H)', 'Germany')

# Transform team names
X.loc[:, 'Home Team'] = le.transform(X['Home Team'])
X.loc[:, 'Away Team'] = le.transform(X['Away Team'])
X_goals.loc[:, 'Home Team'] = le.transform(X_goals['Home Team'])
X_goals.loc[:, 'Away Team'] = le.transform(X_goals['Away Team'])


# Split the data and train models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_goals_train, X_goals_test, y_home_goals_train, y_home_goals_test, y_away_goals_train, y_away_goals_test = train_test_split(X_goals, y_home_goals, y_away_goals, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

model_home_goals = RandomForestRegressor(n_estimators=100, random_state=42)
model_home_goals.fit(X_goals_train, y_home_goals_train)

model_away_goals = RandomForestRegressor(n_estimators=100, random_state=42)
model_away_goals.fit(X_goals_train, y_away_goals_train)


def predict_match(home_team, away_team):
    home_team = 'Germany' if home_team == 'Germany (H)' else home_team
    away_team = 'Germany' if away_team == 'Germany (H)' else away_team
    try:
        home_encoded = le.transform([home_team])[0]
        away_encoded = le.transform([away_team])[0]

        # Create a DataFrame with named features
        match_features = pd.DataFrame({'Home Team': [home_encoded], 'Away Team': [away_encoded]})

        # Predict outcome probabilities
        probs = model.predict_proba(match_features)[0]

        # Predict goals
        home_goals = model_home_goals.predict(match_features)[0]
        away_goals = model_away_goals.predict(match_features)[0]

        # Adjust goals based on outcome probabilities
        home_goals *= (probs[0] + probs[2] / 2)  # Home win + half of draw prob
        away_goals *= (probs[1] + probs[2] / 2)  # Away win + half of draw prob

        # Round goals to nearest integer
        home_goals = round(home_goals)
        away_goals = round(away_goals)

        # Determine winner
        if home_goals > away_goals:
            winner = home_team
        elif home_goals < away_goals:
            winner = away_team
        else:
            winner = np.random.choice([home_team, away_team])

        return home_goals, away_goals, winner

    except ValueError:
        # If a team is not in the training data, use a random prediction
        home_goals = np.random.randint(0, 4)
        away_goals = np.random.randint(0, 4)
        winner = home_team if home_goals > away_goals else away_team if away_goals > home_goals else np.random.choice(
            [home_team, away_team])
        return home_goals, away_goals, winner


# Initialize group tables
def initialize_group_tables(group_tables):
    for group, table in group_tables.items():
        for _, row in table.iterrows():
            row['Pld'] = 0
            row['W'] = 0
            row['D'] = 0
            row['L'] = 0
            row['GF'] = 0
            row['GA'] = 0
            row['GD'] = 0
            row['Pts'] = 0
    return group_tables

group_tables = initialize_group_tables(group_tables)


def update_group_table(group, home_team, away_team, home_goals, away_goals):
    table = group_tables[group]

    for team, goals_for, goals_against in [(home_team, home_goals, away_goals), (away_team, away_goals, home_goals)]:
        team_index = table.index[table['Team'] == team]

        table.loc[team_index, 'Pld'] += 1
        table.loc[team_index, 'GF'] += goals_for
        table.loc[team_index, 'GA'] += goals_against
        table.loc[team_index, 'GD'] = table.loc[team_index, 'GF'] - table.loc[team_index, 'GA']

        if goals_for > goals_against:
            table.loc[team_index, 'W'] += 1
            table.loc[team_index, 'Pts'] += 3
        elif goals_for < goals_against:
            table.loc[team_index, 'L'] += 1
        else:
            table.loc[team_index, 'D'] += 1
            table.loc[team_index, 'Pts'] += 1

    table = table.sort_values(by=['Pts', 'GD', 'GF'], ascending=False)
    table['Pos'] = range(1, len(table) + 1)
    group_tables[group] = table

# Function to display group table
def display_group_table(group):
    table = group_tables[group]
    print(f"\nGroup {group} Standings:")
    print(tabulate(table, headers='keys', tablefmt='pretty', showindex=False))


# Simulate group stage
print("Group Stage Results:")
for group, table in group_tables.items():
    print(f"\nGroup {group}:")
    teams = table['Team'].tolist()
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            home_team, away_team = teams[i], teams[j]
            home_goals, away_goals, _ = predict_match(home_team, away_team)
            update_group_table(group, home_team, away_team, home_goals, away_goals)
            print(f"{home_team} {home_goals} - {away_goals} {away_team}")

    display_group_table(group)


knockout_teams = {}
for group, table in group_tables.items():
    knockout_teams[f"Winner {group}"] = table.iloc[0]['Team']
    knockout_teams[f"Runner-up {group}"] = table.iloc[1]['Team']

# Determine four best third-placed teams
third_place_teams = []
for group, table in group_tables.items():
    third_place_teams.append(table.iloc[2])
third_place_teams.sort(key=lambda x: (x['Pts'], x['GD'], x['GF']), reverse=True)
best_third_place = [team['Team'] for team in third_place_teams[:4]]

knockout_teams.update({
    '3rd Place 1': best_third_place[0],
    '3rd Place 2': best_third_place[1],
    '3rd Place 3': best_third_place[2],
    '3rd Place 4': best_third_place[3]
})
# Function to get team for knockout stage
def get_knockout_team(team_description):
    if team_description in knockout_teams:
        return knockout_teams[team_description]
    elif 'Winner Match' in team_description:
        match_number = team_description.split()[-1]
        return knockout_teams[f"Match {match_number}"]
    elif '3rd Group' in team_description:
        return np.random.choice(best_third_place)
    else:
        return team_description
# Simulate knockout stage
print("\nKnockout Stage Results:")
total_matches = len(fixtures)
knockout_matches = fixtures.iloc[-15:]  # Assuming last 15 matches are knockout stage
match_counter = total_matches - 14  # Start counter for knockout stage

for index, row in knockout_matches.iterrows():
    home_team = get_knockout_team(row['Home Team'])
    away_team = get_knockout_team(row['Away Team'])

    if home_team == away_team:
        continue

    home_goals, away_goals, winner = predict_match(home_team, away_team)

    if home_goals == away_goals:
        print(f"Note: {home_team} vs {away_team} ended in a draw. {winner} chosen as winner after penalties.")

    knockout_teams[f"Match {match_counter}"] = winner

    if match_counter == total_matches:
        round_name = "Final"
    elif match_counter >= total_matches - 2:
        round_name = "Semi-finals"
    elif match_counter >= total_matches - 6:
        round_name = "Quarter-finals"
    else:
        round_name = "Round of 16"

    if round_name in ["Semi-finals", "Final"]:
        print(f"\nHighlighted {round_name}:")
        print(f"{home_team} {home_goals} - {away_goals} {away_team}")
        print(f"Winner: {winner}\n")
    else:
        print(f"{round_name}: {home_team} {home_goals} - {away_goals} {away_team} - Winner: {winner}")

    match_counter += 1

print("\nTournament Winner:", knockout_teams[f"Match {total_matches}"])