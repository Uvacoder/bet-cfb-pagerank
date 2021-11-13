from sportsipy.ncaaf.teams import Teams
from sportsipy.ncaaf.conferences import Conferences
import pickle
from math import exp
from scipy.sparse.linalg import eigs
import numpy as np
import networkx as nx

def normalize_row(l): 
	games_played = [el for el in l if el != 0]
	num_games_played =	len(games_played)
	res = [el/num_games_played for el in l]
	return res

teams = Teams()
for team in teams:
	print('{}: Wins: {}, Losses: {}'.format(team.name, team.wins, team.losses))

Conferences = Conferences()
conferences = {}
for name, d in Conferences.conferences.items():
	conf_team_names = []
	conf_name = d['name']
	team_names = d['teams']
	for key, value in team_names.items():
		conf_team_names.append(value)

	conferences[conf_name] = conf_team_names

reverse_conferences = {}
for key, value in conferences.items():
	for team in value:
		reverse_conferences[team] = key

games_d = {}

games_played_d = {}

omitted_teams = []
for team in teams:
	if team.wins is None:
		omitted_teams.append(team.name)
	# elif team.wins == 0: #this probably shouldnt be allowed, need another way to take care of absorbing states
	elif team.wins + team.losses == 0:
		omitted_teams.append(team.name)

new_teams = list()
for team in teams:
	if team.name not in omitted_teams:
		new_teams.append(team)

teams = new_teams

teams_list = [team.name for team in teams]

teams_list.sort()

teams_dict = {}
for i in range(len(teams_list)):
	teams_dict[teams_list[i]] = i

reverse_teams_dict = {}
for i in range(len(teams_list)):
	reverse_teams_dict[i] = teams_list[i]

num_games_played_d = {team: 0 for team in teams_list}

for team in teams:
	for game in team.schedule:
		boxscore_id = game.boxscore_index
		if boxscore_id not in games_d.keys() and game.opponent_name in teams_list and game.points_for is not None:
			# game_data = Boxscore[game.boxscore_id]
			if game.location == 'Away':
				games_d[boxscore_id] = {'home': game.opponent_name, 'away': team.name, 'neutral': False, 'home_score': game.points_against, 'away_score': game.points_for}
			elif game.location == 'Home':
				games_d[boxscore_id] = {'home': team.name, 'away': game.opponent_name, 'neutral': False, 'home_score': game.points_for, 'away_score': game.points_against}
			elif game.location == 'Neutral':
				games_d[boxscore_id] = {'home': team.name, 'away': game.opponent_name, 'neutral': True, 'home_score': game.points_for, 'away_score': game.points_against}
			# for key, value in games_d[boxscore_id].items():
			# 	print('{}: {}'.format(key, value), end=', ')
			# print('')

with open('cfb_games_d.pickle', 'wb') as f:
	pickle.dump(games_d, f)

# assert len(games_d.keys()) > 0

# with open('cfb_games_d.pickle', 'rb') as f:
# 	games_d = pickle.load(f)

k = 1.05
f = lambda x: (k**x)/(1 + k**x)
home_advantage = 2.5

adjacency_matrix_row = [0 for _ in range(len(teams_list))]
adjacency_matrix = [list(adjacency_matrix_row) for _ in range(len(teams_list))]

copy_games_d = dict(games_d)

for boxscore_id, data in copy_games_d.items():
	if data['neutral'] is False:
		copy_games_d[boxscore_id]['home_score'] -= 2.5

for boxscore_id, data in copy_games_d.items():
	if data['home_score'] > data['away_score']:
		points_for = data['home_score']
		points_against = data['away_score']
		margin = points_for - points_against
		log_score1 = f(margin)
		log_score2 = f(-margin)
		j = teams_dict[data['home']]
		i = teams_dict[data['away']]
		adjacency_matrix[i][j] += log_score1
		adjacency_matrix[j][i] += log_score2
		# print(log_score1, log_score2)

	else: #away team won
		points_for = data['away_score']
		points_against = data['home_score']
		margin = points_for - points_against
		log_score1 = f(margin)
		log_score2 = f(-margin)
		j = teams_dict[data['away']]
		i = teams_dict[data['home']]
		adjacency_matrix[i][j] += log_score1
		adjacency_matrix[j][i] += log_score2

def delete_bad_rows_and_columns(matrix):
	bad_indices = []
	for i in range(len(matrix)):
		if sum(matrix[i]) == 0:
			bad_indices.append(i)
	res = []
	for i in range(len(matrix)):
		if i not in bad_indices:
			res.append([matrix[i][j] for j in range(len(matrix)) if j not in bad_indices])
	return res


adjacency_matrix = delete_bad_rows_and_columns(adjacency_matrix)
M = np.array(adjacency_matrix)
# M = np.transpose(M)
M = [normalize_row(row) for row in M]
M = np.array(M)
# adjacency_matrix = normalized_adjacency_matrix

import csv
with open('cfb.csv', 'w') as f:
	writer = csv.writer(f)
	writer.writerow([reverse_teams_dict[i] for i in range(len(teams_list))])
	for row in adjacency_matrix:
		writer.writerow(row)


# M = np.array(adjacency_matrix)
# M = np.transpose(M)

val, vec = eigs(M, which='LM', k=1)
vec = np.ndarray.flatten(abs(vec))
sorted_indices = vec.argsort()
ranked = [(reverse_teams_dict[i], vec[i]) for i in sorted_indices]
# ranked.reverse()
ratings = {}
f_rating = lambda x: 100*((1/x)**.5)
for i in range(len(ranked)):
	# print('{rank}. {team}: {rating}'.format(rank=i+1, team=ranked[i][0], rating=ranked[i][1]))
	print('{rank}. {team}: {rating}'.format(rank=i+1, team=ranked[i][0], rating=round(f_rating(ranked[i][1]), 2)))
	ratings[ranked[i][0]] = f_rating(ranked[i][1])

with open('cfb_ratings.csv', 'w') as f:
	writer = csv.writer(f)
	for key, value in ratings.items():
		writer.writerow([key, value])


with open('games.csv', 'w') as f:
	writer = csv.writer(f)
	writer.writerow(['homeRating', 'awayRating', 'homeConference', 'awayConference', 'neutral', 'margin'])
	for boxscore_id, game in games_d.items():
		# print(ratings[game['home']])
		# print(reverse_conferences[game['home']])
		# print(int(game['neutral']))
		# print(game['home_score'] - game['away_score'])
		writer.writerow([ratings[game['home']], ratings[game['away']], reverse_conferences[game['home']], reverse_conferences[game['away']], int(game['neutral']), game['home_score'] - game['away_score']])


import pandas as pd

future_games = {}
for team in teams:
	for game in team.schedule:
		if game.points_for is None and game.opponent_name in teams_list:
			if game.location == 'Home':
				future_games[game.boxscore_index] = {'team': team.name, 'teamRating':ratings[team.name], 'opponent': game.opponent_name, 'opponentRating': ratings[game.opponent_name], 'location': game.location}
			else:# game.location == 'Away':
				future_games[game.boxscore_index] = {'team': game.opponent_name, 'teamRating':ratings[game.opponent_name], 'opponent': team.name, 'opponentRating': ratings[team.name], 'location': game.location}
pd.DataFrame.from_dict(data=future_games, orient='index').to_csv('future_games.csv', header=['team', 'teamRating', 'opponent', 'opponentRating', 'location'])


