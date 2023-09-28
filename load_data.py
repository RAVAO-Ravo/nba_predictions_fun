#!/bin/python3
#-*- coding:utf-8 -*-


import pandas as pd


def load_dataset(file: str) -> pd.DataFrame:
	"""
	Charge un ensemble de données à partir d'un fichier CSV, effectue des transformations et renvoie un DataFrame propre.

	Args:
		file (str): Chemin vers le fichier CSV contenant les données.

	Returns:
		pd.DataFrame: Un DataFrame contenant les données nettoyées et prêtes à être utilisées.
	"""
	
	# Charger les données à partir du fichier CSV
	data = pd.read_csv(filepath_or_buffer=file)
	
	# Les colonnes à supprimer
	to_drop = ["GAME_ID", "GAME_STATUS_TEXT", "TEAM_ID_home", "TEAM_ID_away"]
	
	# Les nouvelles colonnes à définir
	new_columns = ["Date",
				   "Home Team",
				   "Away Team",
				   "season",
				   "pts_h",
				   "fg_h",
				   "ft_h",
				   "fg3_h",
				   "ast_h",
				   "reb_h",
				   "pts_a",
				   "fg_a",
				   "ft_a",
				   "fg3_a",
				   "ast_a",
				   "reb_a",
				   "score"]
	
	# Supprimer les colonnes indésirables et les valeurs NaN
	data = data.drop(labels=to_drop, axis=1).dropna()
	
	# Renommer les colonnes
	data.columns = new_columns
	
	# Trier les données par date et réinitialiser l'index
	data = data.sort_values(by="Date").reset_index(drop=True)
	
	# Ajouter une colonne "winner" pour indiquer l'équipe gagnante
	data["winner"] = -1

	# Ajouter une colonne "result" pour indiquer la victoire ou pas
	data["result"] = 0
	
	# Calculer le score et déterminer le gagnant
	for i in range(0, data.shape[0], 1):
		data.at[i, "score"] = data.at[i, "pts_h"] - data.at[i, "pts_a"]
		if 0 < data.at[i, "score"]:
			data.at[i, "winner"] = data.at[i, "Home Team"]
			data.at[i, "result"] = 1
		else:
			data.at[i, "winner"] = data.at[i, "Away Team"]
	
	# Supprimer les lignes avec des valeurs manquantes et réinitialiser l'index
	data = data.dropna().reset_index(drop=True)
	
	# Sauvegarder les données nettoyées dans un fichier CSV de sauvegarde
	data.to_csv("backup.csv", index=False)
	
	return data


def backTrackResult(data: pd.DataFrame, n_shift: int, i: int, is_home: bool) -> pd.DataFrame:
	"""
	Calcule les résultats précédents d'une équipe à domicile ou à l'extérieur.

	Cette fonction calcule les résultats précédents d'une équipe en fonction des paramètres spécifiés
	et met à jour les colonnes correspondantes dans le DataFrame.

	Args:
		data (pd.DataFrame): Le DataFrame de données.
		n_shift (int): Le nombre de matchs précédents à considérer pour le calcul.
		i (int): L'indice de la ligne actuelle dans le DataFrame.
		is_home (bool): Indique si l'équipe est à domicile (True) ou à l'extérieur (False). Par défaut, True.

	Returns:
		pd.DataFrame: Le DataFrame mis à jour avec les résultats précédents.

	Example:
		backTrackResult(data, n_shift=3, i=10, is_home=True)
	"""

	# Détermine si l'équipe est à domicile ou à l'extérieur
	where = "Home" if is_home else "Away"
	team = data.at[i, f"{where} Team"]  # Équipe actuelle

	j = i
	cpt = n_shift  # Compteur du nombre de matchs précédents à considérer

	copy_data = data.copy()

	while j != 0 and cpt != 0:
		j -= 1  # Décrémente l'indice pour remonter dans le temps
		# Vérifie si l'équipe actuelle a joué dans le match précédent
		if (copy_data.at[j, "Home Team"] == team) or (copy_data.at[j, "Away Team"] == team):

			# Crée des noms de colonnes pour stocker les statistiques du match précédent
			pts = f"{where}_pts_{cpt}"
			astM = f"{where}_astM_{cpt}" # Make
			fgM = f"{where}_fgM_{cpt}"
			fg3M = f"{where}_fg3M_{cpt}"

			def_ = f"{where}_def_{cpt}"
			astG = f"{where}_astG_{cpt}" # Given
			fgG = f"{where}_fgG_{cpt}"
			fg3G = f"{where}_fg3G_{cpt}"

			reb = f"{where}_reb_{cpt}"
			wr = f"{where}_wr_{cpt}"

			cpt -= 1  # Décrémente le compteur

			# Si l'équipe actuelle est à domicile dans le match précédent
			if copy_data.at[j, "Home Team"] == team:

				# Met à jour les colonnes avec les statistiques du match précédent
				copy_data.at[i, pts] = copy_data.at[j, "pts_h"]
				copy_data.at[i, astM] = copy_data.at[j, "ast_h"]
				copy_data.at[i, fgM] = copy_data.at[j, "fg_h"]
				copy_data.at[i, fg3M] = copy_data.at[j, "fg3_h"]

				copy_data.at[i, def_] = copy_data.at[j, "pts_a"]
				copy_data.at[i, astG] = copy_data.at[j, "ast_a"]
				copy_data.at[i, fgG] = copy_data.at[j, "fg_a"]
				copy_data.at[i, fg3G] = copy_data.at[j, "fg3_a"]

				copy_data.at[i, reb] = copy_data.at[j, "reb_h"] - copy_data.at[j, "reb_a"]
				copy_data.at[i, wr] = 1 if copy_data.at[j, "result"] == 1 else 0

			# Si l'équipe actuelle est à l'extérieur dans le match précédent
			else:

				# Met à jour les colonnes avec les statistiques du match précédent
				copy_data.at[i, pts] = copy_data.at[j, "pts_a"]
				copy_data.at[i, astM] = copy_data.at[j, "ast_a"]
				copy_data.at[i, fgM] = copy_data.at[j, "fg_a"]
				copy_data.at[i, fg3M] = copy_data.at[j, "fg3_a"]

				copy_data.at[i, def_] = copy_data.at[j, "pts_h"]
				copy_data.at[i, astG] = copy_data.at[j, "ast_h"]
				copy_data.at[i, fgG] = copy_data.at[j, "fg_h"]
				copy_data.at[i, fg3G] = copy_data.at[j, "fg3_h"]

				copy_data.at[i, reb] = copy_data.at[j, "reb_a"] - copy_data.at[j, "reb_h"]
				copy_data.at[i, wr] = 1 if copy_data.at[j, "result"] == 0 else 0

	return copy_data


def calculate_previous_results(data: pd.DataFrame, n_shift: int=1) -> pd.DataFrame:
	"""
	Calcule les résultats précédents des équipes dans un DataFrame de données.

	Args:
		data (pd.DataFrame): Le DataFrame de données d'origine.
		n_shift (int, optionnal): Le nombre de matchs précédents à considérer pour le calcul. Par défaut, 1.

	Returns:
		pd.DataFrame: Le DataFrame de données avec les résultats précédents ajoutés.
	"""

	results_data = pd.DataFrame()

	# Parcours des saisons uniques dans la colonne 'season'
	for season in data['season'].unique() :

		# Filtrez le DataFrame pour ne conserver que les lignes de cette saison
		season_data = data[data['season'] == season].copy().reset_index(drop=True)

		# Parcours des équipes uniques dans le DataFrame
		for i in range(0, season_data.shape[0], 1) :
			# Calcule les résultats précédents pour l'équipe à domicile (is_home=True)
			season_data = backTrackResult(data=season_data, n_shift=n_shift, i=i, is_home=True)
			# Calcule les résultats précédents pour l'équipe à l'extérieur (is_home=False)
			season_data = backTrackResult(data=season_data, n_shift=n_shift, i=i, is_home=False)

		# Concatène les données de la saison actuelle au DataFrame des résultats
		results_data = pd.concat([results_data, season_data], axis=0, ignore_index=True)

	# Supprime les lignes contenant des valeurs manquantes et réinitialise l'index
	return results_data.dropna().reset_index(drop=True)


def calculate_team_statistics(data: pd.DataFrame) -> pd.DataFrame:
	"""
	Calcule diverses statistiques pour les équipes à domicile et à l'extérieur à partir d'un DataFrame de données de matchs.

	Args:
		data (pd.DataFrame): Le DataFrame de données d'origine.

	Returns:
		pd.DataFrame: Un DataFrame contenant des statistiques pour les équipes à domicile et à l'extérieur.
	"""
	
	# Colonnes à utiliser pour le calcul des moyennes et des statistiques
	wheres = ["Home", "Away"]
	stats = ["pts", "astM", "fgM", "fg3M", "def", "astG", "fgG", "fg3G", "reb", "wr"]

	final_features = pd.DataFrame()

	for where in wheres:
		for stat in stats:
			
			# Sélectionne les colonnes correspondant aux caractéristiques
			dataf = data.filter(regex=f"^{where}_{stat}")

			# Calcul des statistiques de base
			mod_feature = dataf.mode(axis=1)[0].to_frame(name=f"{where}_{stat}_mode")
			mean_feature = dataf.mean(axis=1).to_frame(name=f"{where}_{stat}_mean")
			std_feature = dataf.std(axis=1).to_frame(name=f"{where}_{stat}_std")
			var_feature = dataf.var(axis=1).to_frame(name=f"{where}_{stat}_var")
			
			# Ajout de statistiques supplémentaires si la statistique n'est pas "wr"
			if stat != "wr":
				q25_feature = dataf.quantile(q=0.25, axis=1).to_frame(name=f"{where}_{stat}_q25")
				med_feature = dataf.median(axis=1).to_frame(name=f"{where}_{stat}_median")
				q75_feature = dataf.quantile(q=0.75, axis=1).to_frame(name=f"{where}_{stat}_q75")
				min_feature = dataf.min(axis=1).to_frame(name=f"{where}_{stat}_min")
				max_feature = dataf.max(axis=1).to_frame(name=f"{where}_{stat}_max")
				skw_feature = dataf.skew(axis=1).to_frame(name=f"{where}_{stat}_skw")
				kur_feature = dataf.kurtosis(axis=1).to_frame(name=f"{where}_{stat}_kur")
				iqr_feature = pd.DataFrame((q75_feature.values-q25_feature.values), columns=[f"{where}_{stat}_iqr"])
				
				# Concaténation des caractéristiques au DataFrame final
				final_features = pd.concat([final_features,
											mod_feature,
											mean_feature,
											std_feature,
											var_feature,
											q25_feature,
											q75_feature,
											iqr_feature,
											med_feature,
											min_feature,
											max_feature,
											skw_feature,
											kur_feature], axis=1)
			
			else:
				final_features = pd.concat([final_features,
											mod_feature,
											mean_feature,
											std_feature,
											var_feature], axis=1)

	# Suppression des colonnes NaN et des colonnes booléennes
	bool_columns = final_features.select_dtypes(include=['bool']).columns

	return final_features.dropna(axis=1).drop(labels=bool_columns, axis=1)


def correlation(data: pd.DataFrame, target_column: str, threshold: float=0.1) -> pd.DataFrame:
	"""
	Calcule la corrélation entre les caractéristiques et une variable cible, puis renvoie les caractéristiques 
	dont la corrélation est supérieure ou égale au seuil spécifié.

	Args:
		data (pd.DataFrame): Le DataFrame contenant les caractéristiques et la variable cible.
		target_column (str): Le nom de la variable cible.
		threshold (float, optional): Le seuil de corrélation à partir duquel sélectionner les caractéristiques.
			Par défaut, 0.1.

	Returns:
		pd.DataFrame: Un DataFrame contenant les caractéristiques dont la corrélation avec la variable cible 
		est supérieure ou égale au seuil.

	Example:
		high_corr_features = correlation(data, 'target_variable', threshold=0.2)
	"""
	
	# Créez une nouvelle DataFrame contenant uniquement les caractéristiques et la variable cible
	data_subset = data.drop(columns=[target_column])

	# Calculez la matrice de corrélation entre les caractéristiques et la variable cible
	correlation_matrix = data_subset.corrwith(data[target_column], method="spearman")

	# Créez une DataFrame avec la matrice de corrélation
	correlation_df = pd.DataFrame({'Correlation': correlation_matrix})

	# Triez la DataFrame par ordre décroissant de corrélation absolue
	correlation_df = correlation_df.abs().sort_values(by='Correlation', ascending=False)

	# Sélectionnez les caractéristiques ayant une corrélation élevée par rapport au seuil
	high_correlation_features = correlation_df[correlation_df['Correlation'] >= threshold]

	return high_correlation_features