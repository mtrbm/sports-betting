

"""**Data I/O**"""
import os
os.environ['PATH'] += os.pathsep+'C:/Program Files (x86)/Graphviz2.38/bin'
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

rankings = pd.read_csv('C:/Users/fwdw9/Desktop/node js/worldcup/fifa_ranking.csv')

rankings = rankings.loc[:,['rank', 'country_full', 'country_abrv', 'cur_year_avg_weighted', 'rank_date',
                           'two_year_ago_weighted', 'three_year_ago_weighted']]
rankings = rankings.replace({"IR Iran": "Iran"})
rankings['weighted_points'] = rankings['cur_year_avg_weighted'] + rankings['two_year_ago_weighted'] + rankings['three_year_ago_weighted']
rankings['rank_date'] = pd.to_datetime(rankings['rank_date'])

matches = pd.read_csv('C:/Users/fwdw9/Desktop/node js/worldcup/results.csv')
matches = matches.replace({'Germany DR': 'Germany', 'China' : 'China PR'})
matches['date'] = pd.to_datetime(matches['date'])

world_cup = pd.read_csv('C:/Users/fwdw9/Desktop/node js/worldcup/World Cup 2018 Dataset.csv')
world_cup = world_cup.loc[:, ['Team', 'Group', 'First match \nagainst', 'Second match\n against', 'Third match\n against']]
world_cup = world_cup.dropna(how='all')
world_cup = world_cup.replace({'IRAN' : 'Iran',
                               'Costarica': 'Costa Rica',
                               'Porugal': 'Portugal',
                               'Columbia': 'Colombia',
                               'Korea': 'Korea Republic'})

world_cup = world_cup.set_index('Team')



"""**Feature extraction**"""



matches.merge(rankings,
                    left_on=['date', 'home_team'],
                    right_on=['rank_date', 'country_full'])



# I want to have the ranks for every day
rankings = rankings.set_index(['rank_date'])\
            .groupby(['country_full'], group_keys=False)\
            .resample('D').first()\
            .fillna(method='ffill')\
            .reset_index()

# join the ranks
# rank_home, rank_away 구별되게 column 만들어줌
matches = matches.merge(rankings,
                        left_on=['date', 'home_team'],
                        right_on=['rank_date', 'country_full'])
matches = matches.merge(rankings,
                        left_on=['date', 'away_team'],
                        right_on=['rank_date', 'country_full'],
                        suffixes=('_home', '_away'))

# feature generation
matches['rank_difference'] = matches['rank_home'] - matches['rank_away']
matches['average_rank'] = (matches['rank_home'] + matches['rank_away'])/2
matches['point_difference'] = matches['weighted_points_home'] - matches['weighted_points_away']
matches['score_difference'] = matches['home_score'] - matches['away_score']
matches['is_won'] = matches['score_difference'] > 0 # take draw as lost
matches['is_stake'] = matches['tournament'] != 'Friendly'

# I tried earlier rest days but it did not turn to be useful
max_rest = 30
matches['rest_days'] = matches.groupby('home_team')['date'].diff().dt.days.clip(0,max_rest).fillna(max_rest)

# I tried earlier the team as well but that did not make a difference either
matches['wc_participant'] = matches['home_team'] * matches['home_team'].isin(world_cup.index.tolist())
matches['wc_participant'] = matches['wc_participant'].replace({'':'Other'})
matches = matches.join(pd.get_dummies(matches['wc_participant']))

matches[matches['wc_participant']]

"""**Modeling**"""

from sklearn import linear_model
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

X, y = matches.loc[:, ['average_rank', 'rank_difference', 'point_difference', 'is_stake']], matches['is_won']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

logreg = linear_model.LogisticRegression(C=1e-5)
features = PolynomialFeatures(degree=2)
model = Pipeline([
    ('polynomial_features', features),
    ('logistic_regression', logreg)
])
model = model.fit(X_train, y_train)



features = ['average_rank', 'rank_difference', 'point_difference']
wrongs = y_test != model.predict(X_test)



"""**World Cup simulation**"""

# let's define a small margin when we safer to predict draw then win
margin = 0.05

# let's define the rankings at the time of the World Cup
world_cup_rankings = rankings.loc[(rankings['rank_date'] == rankings['rank_date'].max()) &
                                rankings['country_full'].isin(world_cup.index.unique())]
world_cup_rankings = world_cup_rankings.set_index(['country_full'])
from itertools import combinations

opponents = ['First match \nagainst', 'Second match\n against', 'Third match\n against']

world_cup['points'] = 0
world_cup['total_prob'] = 0
print(world_cup)

def initialize_world_cup():
    global world_cup
    world_cup['points'] = 0
    world_cup['total_prob'] = 0
    return world_cup.copy()

def initialize_world_cup_rankings():
    global world_cup_rankings
    return world_cup_rankings.copy()

def get_home_win_prob(home, away):
    global world_cup_rankings, model
    world_cup_ranking = world_cup_rankings.copy()

    row = pd.DataFrame(np.array([[np.nan, np.nan, np.nan, True]]), columns=X_test.columns)
    home_rank = world_cup_ranking.loc[home, 'rank']
    home_points = world_cup_ranking.loc[home, 'weighted_points']
    opp_rank = world_cup_ranking.loc[away, 'rank']
    opp_points = world_cup_ranking.loc[away, 'weighted_points']
    row['average_rank'] = (home_rank + opp_rank) / 2
    row['rank_difference'] = home_rank - opp_rank
    row['point_difference'] = home_points - opp_points

    home_win_prob = model.predict_proba(row)[:, 1][0]
    return home_win_prob

def get_actual_result(home_win_prob):
    if home_win_prob > 0.55:
        return "home"
    elif home_win_prob < 0.45:
        return "away"
    else:
        return "draw"

def get_all_matches(world_cup):
    all_matches = []
    for group in set(world_cup['Group']):
        for home, away in combinations(world_cup.query(f'Group == "{group}"').index, 2):
            all_matches.append((home, away))
    return all_matches
def world_cup_group_match():
    global world_cup
    group_match = world_cup.copy()
    Group_result = []
    for group in set(world_cup['Group']):
        group_result = []
        group_result.append(f'___Starting group {group}:___')
        for home, away in combinations(group_match.query('Group == "{}"'.format(group)).index, 2):
            result_line = f'{home} vs {away}:'
            home_win_prob = get_home_win_prob(home, away)
            result_line += f'home_win_prob : {home_win_prob} ' 

            group_match.loc[home, 'total_prob'] += home_win_prob
            group_match.loc[away, 'total_prob'] += home_win_prob

            points = 0

            # [이긴 국가, 포인트] 리턴하기
            if home_win_prob <= 0.5 - margin:
                group_match.loc[away, 'points'] += 3
            if home_win_prob > 0.5 - margin:
                points = 1
            if home_win_prob >= 0.5 + margin:
                points = 3
                group_match.loc[home, 'points'] += 3
            if points == 1:
                group_match.loc[home, 'points'] += 1
                group_match.loc[away, 'points'] += 1
            
            group_result.append(result_line)    
        
        Group_result.append('\n'.join(group_result))
    sorted_groups = group_match.groupby('Group').apply(lambda x: x.sort_values(by='points', ascending=False))
    print(group_match)
    return sorted_groups

"""**Single-elimination rounds**"""
# def single_elimination_rounds():
#     global world_cup
#     single_match = world_cup.copy()
#     pairing = [0,3,4,7,8,11,12,15,1,2,5,6,9,10,13,14]
#     print('world_cup_check', single_match)
#     single_match = single_match.sort_values(by=['Group', 'points', 'total_prob'], ascending=False).reset_index()
#     print('world_cup_check', single_match)

#     next_round_wc = single_match.groupby('Group').nth([0, 1]) # select the top 2
#     next_round_wc = next_round_wc.reset_index()
#     next_round_wc = next_round_wc.loc[pairing]
#     next_round_wc = next_round_wc.set_index('Team')

#     finals = ['round_of_16', 'quarterfinal', 'semifinal', 'final']
#     rounds_data = {}
#     labels = list()
#     odds = list()

#     for f in finals:
#         print("___Starting of the {}___".format(f))
#         iterations = int(len(next_round_wc) / 2)
#         winners = []
#         data = []
        
#         for i in range(iterations):
#             home = next_round_wc.index[i*2]
#             away = next_round_wc.index[i*2+1]
#             print("{} vs. {}: ".format(home,
#                                     away),
#                                     end='')
#             row = pd.DataFrame(np.array([[np.nan, np.nan, np.nan, True]]), columns=X_test.columns)
#             home_rank = world_cup_rankings.loc[home, 'rank']
#             home_points = world_cup_rankings.loc[home, 'weighted_points']
#             opp_rank = world_cup_rankings.loc[away, 'rank']
#             opp_points = world_cup_rankings.loc[away, 'weighted_points']
#             row['average_rank'] = (home_rank + opp_rank) / 2
#             row['rank_difference'] = home_rank - opp_rank
#             row['point_difference'] = home_points - opp_points

#             home_win_prob = model.predict_proba(row)[:,1][0]
#             if model.predict_proba(row)[:,1] <= 0.5:
#                 print("{0} wins with probability {1:.2f}".format(away, 1-home_win_prob))
#                 winners.append(away)
#             else:
#                 print("{0} wins with probability {1:.2f}".format(home, home_win_prob))
#                 winners.append(home)

#             data.append({
#                 'home': world_cup_rankings.loc[home, 'country_abrv'],
#                 'away': world_cup_rankings.loc[away, 'country_abrv'],
#                 'home_odds': 1/home_win_prob,
#                 'away_odds': 1/(1-home_win_prob)
#             })
            

#         next_round_wc = next_round_wc.loc[winners]
#         df = pd.DataFrame(data)
#         rounds_data[f] = df
#         print("\n")

#     single_match.groupby('Group').nth([0, 1])
#     print('labels = ', labels)
#     print('odds = ', odds)
#     return rounds_data

