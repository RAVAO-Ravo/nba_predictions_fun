#!/bin/python3
#-*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import random as rd
import seaborn as sns
import typing as tp
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import learning_curve


def plot_1d(features: pd.DataFrame, target: pd.Series) -> None:
	"""
	Réduit un DataFrame à une seule dimension en utilisant PCA et trace deux KDE plots en fonction des labels.

	Args:
		dataframe (pd.DataFrame): Le DataFrame contenant les features.
		labels (pd.Series): Une série contenant les labels correspondants aux features.

	Returns:
		None
	"""
	# Réduire le DataFrame à une dimension en utilisant PCA
	pca = PCA(n_components=1, random_state=42)
	df_1d = pca.fit_transform(features)

	# Créez un DataFrame à partir de la réduction à une dimension et des labels
	df_1d = pd.DataFrame(df_1d, columns=['Dimension 1'])
	df_1d['Labels'] = target.reset_index(drop=True)

	# Tracez deux KDE plots en fonction des labels
	plt.figure(figsize=(8, 6))  # Ajustez la taille du plot si nécessaire
	sns.kdeplot(data=df_1d[df_1d['Labels'] == 0]['Dimension 1'], label='Défaites', shade=True)
	sns.kdeplot(data=df_1d[df_1d['Labels'] == 1]['Dimension 1'], label='Victoires', shade=True)
	plt.title("KDE Plot - Dimension 1")
	plt.xlabel("Valeur")
	plt.ylabel("Densité")
	plt.legend()
	plt.show()


def plot_2d(X: pd.DataFrame, y: pd.Series, method: str='pca') -> None:
	"""
	Effectue une analyse de réduction de dimension sur les données et génère un scatterplot avec légende.

	Cette fonction prend un DataFrame de données X et une série d'étiquettes y, effectue une réduction de dimension
	en 2 dimensions en utilisant la méthode spécifiée (PCA ou t-SNE), et affiche un scatterplot coloré en fonction
	des étiquettes.

	Args:
		X (pd.DataFrame): Les données à utiliser pour la réduction de dimension.
		y (pd.Series): Les étiquettes correspondantes.
		method (str, optional): La méthode de réduction de dimension à utiliser ('pca' ou 'tsne'). Par défaut, 'pca'.

	Returns:
		None
	"""
	
	# Réduction de dimension en utilisant PCA
	if method == "pca":
		reduction_result = PCA(n_components=2, random_state=42).fit_transform(X)

	# Réduction de dimension en utilisant t-SNE
	elif method == "tsne":
		reduction_result = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=42).fit_transform(X)

	else:
		raise ValueError("La méthode spécifiée doit être 'pca' ou 'tsne'.")

	# Extraction des coordonnées x et z
	x = [i for i, j in reduction_result]
	z = [j for i, j in reduction_result]

	# Détermination des couleurs en fonction des étiquettes
	colors: tp.List[str] = ["red" if label == 1 else "blue" for label in y.tolist()]

	# Affichage du scatterplot avec légende
	plt.figure(figsize=(12, 8))
	for label, color in {"Défaites": "blue", "Victoires": "red"}.items():
		plt.scatter([], [], c=color, label=label)
	plt.scatter(x, z, c=colors, marker='o', edgecolors="black")
	plt.xlabel("Features 1")
	plt.ylabel("Features 2")
	plt.legend(loc='best')
	plt.show()



def generate_random_boolean_list(df: pd.DataFrame, n_true: int, random_state: int=42) -> tp.List[bool]:
	"""
	Génère une liste de booléens aléatoire de même taille que le nombre d'éléments dans le DataFrame,
	avec n_true valeurs True et le reste False. Utilise une graine aléatoire pour la reproductibilité.

	Args:
		df (pd.DataFrame): Le DataFrame pour lequel vous souhaitez générer la liste.
		n_true (int): Le nombre de True dans la liste.
		random_state (int, optional): Graine aléatoire pour la reproductibilité. Par défaut, None.

	Returns:
		List[bool]: Une liste de booléens.
	"""
	# Fixez la graine aléatoire pour la reproductibilité si une graine est fournie
	if random_state is not None:
		rd.seed(random_state)

	total_elements = len(df)
	if n_true > total_elements:
		raise ValueError("Le nombre de True ne peut pas être supérieur au nombre total d'éléments dans le DataFrame.")
	
	# Générez la liste avec n_true True et le reste False, puis mélangez-la
	boolean_list = [True] * n_true + [False] * (total_elements - n_true)
	rd.shuffle(boolean_list)
	
	return boolean_list


def plot_3d(features: pd.DataFrame, target: pd.Series, n: int=500, random_state: int=42) -> None:
	"""
	Effectue une réduction de dimensionnalité à 3 composantes principales (PCA) 
	et affiche les données en 3D de manière interactive avec Plotly.

	Args:
		features (pd.DataFrame): Le DataFrame contenant les caractéristiques.
		target (pd.DataFrame): Le DataFrame contenant les étiquettes.
		n (int, optionnal): Le nombre d'éléments.
		random_state (int, optional): Graine aléatoire pour la reproductibilité. Par défaut, 42.

	Returns:
		None
	"""

	# Nombre d'éléments à sélectionner
	selected = generate_random_boolean_list(df=target, n_true=n, random_state=random_state)

	# Réduction de dimension à 3 composantes principales
	X_reduced = PCA(n_components=3, random_state=random_state).fit_transform(features[selected])

	# Créez un DataFrame avec les composantes principales réduites et les étiquettes
	df_reduced = pd.DataFrame(data=X_reduced, columns=['PC1', 'PC2', 'PC3'])
	df_reduced['Label'] = target[selected].reset_index(drop=True)

	# Utilisez Plotly Express pour créer un graphique 3D interactif
	fig = px.scatter_3d(df_reduced, x='PC1', y='PC2', z='PC3', color='Label')

	# Personnalisez le graphique
	fig.update_layout(scene=dict(
						xaxis_title='Principal Component 1',
						yaxis_title='Principal Component 2',
						zaxis_title='Principal Component 3'
					))

	# Affichez le graphique interactif dans le notebook
	fig.show()


def plot_confusion_matrix(y_true: tp.Any, y_pred: tp.Any) -> None:
	"""
	Affiche une matrice de confusion sous forme de heatmap.

	Args:
		y_true (Any): Les vraies étiquettes.
		y_pred (Any): Les étiquettes prédites.

	Returns:
		None
	"""
	# Calculer la matrice de confusion
	conf_matrix = confusion_matrix(y_true, y_pred)

	# Créer un heatmap de la matrice de confusion
	plt.figure(figsize=(12, 8))
	sns.set(font_scale=1.2)  # Réglez la taille de la police
	sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
				xticklabels=['Défaites', 'Victoires'], yticklabels=['Défaites', 'Victoires'])
	plt.xlabel('Prédictions')
	plt.ylabel('Réalités')
	plt.title('Matrice de Confusion')
	plt.show()


def plot_roc_auc(y_true: tp.Any, y_pred: tp.Any) -> None:
	"""
	Affiche la courbe ROC (Receiver Operating Characteristic) avec l'AUC (Area Under the Curve).

	Args:
		y_true (Any): Les vraies étiquettes.
		y_pred (Any): Les scores prédits (probabilités).

	Returns:
		None
	"""
	# Calculer la courbe ROC et l'AUC
	fpr, tpr, thresholds = roc_curve(y_true, y_pred)
	roc_auc = roc_auc_score(y_true, y_pred)

	# Tracer la courbe ROC
	plt.figure(figsize=(12, 8))
	plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"Courbe ROC (AUC = {roc_auc:0.2f})")
	plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('Taux de Faux Positifs (FPR)')
	plt.ylabel('Taux de Vrais Positifs (TPR)')
	plt.title('Courbe ROC')
	plt.legend(loc='lower right')
	plt.show()


def evaluate_models(models: dict, X_train: tp.Any, y_train: tp.Any, X_test: tp.Any, y_test: tp.Any) -> dict:
	"""
	Évalue plusieurs modèles de machine learning et génère un rapport de classification pour chacun.

	Args :
		models (dict): Un dictionnaire contenant les modèles à évaluer avec leur nom en tant que clé et le modèle en tant que valeur.
		X_train (Any): Les données d'entraînement.
		y_train (Any): Les étiquettes d'entraînement.
		X_test (Any): Les données de test.
		y_test (Any): Les étiquettes de test.

	Returns :
		dict: Un dictionnaire contenant les rapports de classification pour chaque modèle.
	"""

	for model_name, model in models.items():
		print(f"Rapport de classification pour {model_name} :\n")
		
		# Entraînement du modèle
		model.fit(X_train, y_train)

		# Prédictions sur les données de test
		y_pred = model.predict(X_test)

		# Affichage du rapport de classification
		print(classification_report(y_test, y_pred))

		# Affichage de la matrice de confusion
		plot_confusion_matrix(y_test, y_pred)

		# Affichage de la courbe ROC et de l'AUC
		plot_roc_auc(y_test, y_pred)


def plot_learning_curve(estimator: BaseEstimator,
						X: tp.Any,
						y: tp.Any,
						cv: tp.Optional[int]=None,
						train_sizes: tp.Optional[np.ndarray]=np.linspace(0.1, 1.0, 10),
						scoring: tp.Optional[str]='accuracy',
						title: tp.Optional[str]="Learning Curve") -> None:
	"""
	Affiche la courbe d'apprentissage d'un modèle.

	Args:
		estimator (BaseEstimator): Le modèle à évaluer.
		X (Any): Les données d'entraînement.
		y (Any): Les étiquettes d'entraînement.
		cv (Optional[int]): La stratégie de validation croisée. Si None, utilise la validation croisée par défaut de 5 plis.
		train_sizes (Optional[np.ndarray]): Les tailles des ensembles d'entraînement à évaluer.
		scoring (Optional[str]): La métrique de score à utiliser.
		title (Optional[str]): Le titre du graphique.

	Returns:
		None
	"""
	train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, train_sizes=train_sizes, scoring=scoring)

	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)

	plt.figure(figsize=(12, 8))
	plt.title(title)
	plt.xlabel("Nombre d'échantillons d'entraînement")
	plt.ylabel(scoring)

	plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
	plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")

	plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Score d'entraînement")
	plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Score de validation")

	plt.legend(loc="best")
	plt.grid()
	plt.show()


def plot_decision_boundary(model: BaseEstimator, X: tp.Any, y: tp.Any) -> None:
	"""
	Affiche la frontière de décision d'un modèle d'apprentissage supervisé.

	Args:
		model (BaseEstimator): Le modèle d'apprentissage supervisé déjà entraîné.
		X (Any): Les caractéristiques (features) de l'ensemble de données.
		y (Any): Les labels correspondants.

	Returns:
		None
	"""

	# Vérifie si le modèle est un classificateur binaire (2 classes)
	if len(np.unique(y)) != 2:
		raise ValueError("Ce plot est conçu pour les classificateurs binaires (2 classes).")

	# Crée une grille de points pour afficher la frontière de décision
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

	# Prédit les classes pour chaque point de la grille
	Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)

	# Affiche la frontière de décision
	plt.contourf(xx, yy, Z, alpha=0.4)
	plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', edgecolor='k')
	plt.xlabel("Feature 1")
	plt.ylabel("Feature 2")


def plot_decision_boundaries(models: tp.Dict[str, BaseEstimator], X: np.ndarray, y: np.ndarray) -> None:
	"""
	Trace les frontières de décision pour une liste de modèles après une réduction de dimension PCA.

	Args:
		models (Dict[str, BaseEstimator]): Un dictionnaire contenant les modèles à évaluer avec leur nom en tant que clé
			et le modèle en tant que valeur.
		X (np.ndarray): Les données d'entrée.
		y (np.ndarray): Les étiquettes correspondantes.

	Returns:
		None
	"""

	# Réduction de dimension en utilisant PCA
	x_pca = PCA(n_components=2, random_state=42).fit_transform(X)

	# Configuration de la disposition des sous-graphiques
	num_models = len(models)
	num_cols = 2  # Nombre de colonnes dans la disposition
	num_rows = (num_models + 1) // num_cols  # Calcul du nombre de lignes

	# Création de sous-graphiques
	plt.figure(figsize=(30, 10))
	for i, (name_mod, model) in enumerate(models.items()):
		
		plt.subplot(num_rows, num_cols, i+1)
		
		# Ajustement du modèle aux données PCA
		mod = model
		mod.fit(x_pca, y)
		
		# Affichage des frontières de décision
		plot_decision_boundary(mod, x_pca, y)
		plt.title(f"Frontière de Décision {name_mod}")

	plt.show()