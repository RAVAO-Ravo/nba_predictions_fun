#!/bin/python3
#-*- coding:utf-8 -*-


import numpy as np
import pandas as pd
import typing as tp
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import IsolationForest
from twinning import twin


def preprocess_data(X: tp.Any, y: tp.Any, contamination: float="auto", random_state: int=42) -> tp.Tuple[tp.Any, tp.Any]:
	"""
	Prétraite les données en retirant les outliers avec Isolation Forest et en équilibrant les classes.

	Args :
	- X (Any) : Les données à prétraiter.
	- y (Any) : Les étiquettes des données.
	- contamination (float) : Taux de contamination pour Isolation Forest (par défaut 'auto').
	- random_state (int) : Seed pour la reproductibilité (par défaut 42).

	Returns :
	- X (Any) : Les données prétraitées.
	- y (Any) : Les étiquettes des données prétraitées.
	"""

	# Retirez les outliers à l'aide de l'Isolation Forest
	iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
	outlier_labels = iso_forest.fit_predict(X=X, y=y)
	X = X[outlier_labels == 1]
	y = y[outlier_labels == 1]

	# Équilibrez les classes en sous-échantillonnant la classe majoritaire
	X, y = RandomUnderSampler(sampling_strategy="auto", random_state=random_state, replacement=False).fit_resample(X=X, y=y)

	return X, y


def train_test_split_twin(features: tp.Any, target: tp.Any, test_size: tp.Union[float, int]=0.2) -> tuple:
	"""
	Divise les données en ensembles d'entraînement et de test en utilisant la méthode twinning.

	Cette fonction prend en entrée les caractéristiques (features) et la variable cible (target), et divise les données
	en ensembles d'entraînement et de test en utilisant la méthode twinning. Le paramètre test_size
	spécifie la taille du jeu de test en pourcentage ou en nombre d'échantillons.

	Args:
		features (Any): Les caractéristiques à diviser en ensembles d'entraînement et de test.
		target (Any): La variable cible correspondant aux caractéristiques.
		test_size (Union[float, int], optional): La taille du jeu de test en pourcentage (float) ou en nombre d'échantillons (int).
			Par défaut, 0.25.

	Returns:
		Tuple : Un tuple contenant les ensembles d'entraînement et de test
		pour les caractéristiques et la variable cible (x_train, x_test, y_train, y_test).
	"""

	# Vérifiez si les données sont des DataFrames et convertissez-les en tableaux NumPy si nécessaire
	if isinstance(features, pd.DataFrame):
		X = features.values
	if isinstance(target, pd.Series):
		y = target.values
	
	# Calculez la taille du jeu de test en fonction de test_size (peut être un flottant ou un entier)
	if isinstance(test_size, int):
		r = int(1//(test_size/X.shape[0]))
	elif isinstance(test_size, float):
		r = int(1//test_size)
	else:
		raise ValueError("test_size doit être un nombre flottant ou un entier.")
	
	# Utilisez la fonction twin pour obtenir les indices du jeu de test
	test_indices = twin(X, r=r, u1=0)
	
	# Divisez les données en ensembles d'entraînement et de test
	x_train, x_test = np.delete(X, test_indices, axis=0), X[test_indices]
	y_train, y_test = np.delete(y, test_indices, axis=0), y[test_indices]
	
	# Si les données d'origine étaient des DataFrames, renvoyez des DataFrames
	if isinstance(features, pd.DataFrame):
		x_train, x_test = pd.DataFrame(x_train), pd.DataFrame(x_test)
	if isinstance(target, pd.Series):
		y_train, y_test = pd.Series(y_train), pd.Series(y_test)
	
	return x_train, x_test, y_train, y_test