{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "YgyeUA40aslE"
   },
   "source": [
    "# Modified Lab: Feature Selection with Interactive Analysis\n",
    "**Student**: uXmii\n",
    "**Modifications**: \n",
    "1. Uses sklearn's built-in breast cancer dataset\n",
    "2. Added interactive visualizations\n",
    "3. Added feature importance comparison dashboard\n",
    "4. Extended evaluation with cross-validation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Optzx97aahnJ"
   },
   "source": [
    "Feature selection involves picking the set of features that are most relevant to the target variable. This helps in reducing the complexity of your model, as well as minimizing the resources required for training and inference. This has greater effect in production models where you maybe dealing with terabytes of data or serving millions of requests.\n",
    "\n",
    "In this notebook, you will run through the different techniques in performing feature selection on the [Breast Cancer Dataset](http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29). Most of the modules will come from [scikit-learn](https://scikit-learn.org/stable/), one of the most commonly used machine learning libraries. It features various machine learning algorithms and has built-in implementations of different feature selection methods. Using these, you will be able to compare which method works best for this particular dataset."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "ZEnMK4DRNV1O"
   },
   "source": [
    "## Imports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "ZersTw6TH1Zj"
   },
   "outputs": [],
   "source": [
    "# for data processing and manipulation\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "# scikit-learn modules for feature selection and model evaluation\n",
    "from sklearn.datasets import load_breast_cancer\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.feature_selection import RFE, SelectKBest, SelectFromModel, chi2, f_classif\n",
    "from sklearn.model_selection import train_test_split, cross_val_score\n",
    "from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score\n",
    "from sklearn.svm import LinearSVC\n",
    "from sklearn.feature_selection import SelectFromModel\n",
    "from sklearn.preprocessing import StandardScaler, MinMaxScaler\n",
    "\n",
    "# libraries for visualization\n",
    "import seaborn as sns\n",
    "import matplotlib\n",
    "import matplotlib.pyplot as plt\n",
    "import plotly.express as px\n",
    "import plotly.graph_objects as go\n",
    "from plotly.subplots import make_subplots\n",
    "\n",
    "# Display settings\n",
    "plt.style.use('seaborn-v0_8')\n",
    "pd.set_option('display.max_columns', None)\n",
    "\n",
    "print(\"✅ All libraries imported successfully!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "TvMpn0VaazcC"
   },
   "source": [
    "## Load the dataset - MODIFIED VERSION\n",
    "\n",
    "**Modification 1**: Instead of loading from CSV, we're using sklearn's built-in breast cancer dataset for better reproducibility and to demonstrate different data loading techniques."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "DspE2DYYPpRp"
   },
   "outputs": [],
   "source": [
    "# Load the dataset using sklearn's built-in function\n",
    "breast_cancer_data = load_breast_cancer()\n",
    "\n",
    "# Create DataFrame\n",
    "df = pd.DataFrame(breast_cancer_data.data, columns=breast_cancer_data.feature_names)\n",
    "df['diagnosis'] = breast_cancer_data.target\n",
    "\n",
    "# Print dataset information\n",
    "print(\"Dataset Description:\")\n",
    "print(breast_cancer_data.DESCR[:500] + \"...\")\n",
    "print(f\"\\nShape: {df.shape}\")\n",
    "print(f\"Features: {len(breast_cancer_data.feature_names)}\")\n",
    "print(f\"Classes: {breast_cancer_data.target_names}\")\n",
    "\n",
    "# Print datatypes\n",
    "print(\"\\nData types:\")\n",
    "print(df.dtypes.value_counts())\n",
    "\n",
    "# Describe columns\n",
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Preview the dataset\n",
    "print(\"First 5 rows:\")\n",
    "display(df.head())\n",
    "\n",
    "print(\"\\nTarget distribution:\")\n",
    "print(df['diagnosis'].value_counts())\n",
    "print(\"\\nTarget distribution (percentage):\")\n",
    "print(df['diagnosis'].value_counts(normalize=True) * 100)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "fJ9yk-r6bYdZ"
   },
   "source": [
    "## Data Preparation - No Need for Integer Encoding\n",
    "\n",
    "**Modification 2**: Since sklearn's dataset already provides integer targets (0 for malignant, 1 for benign), we don't need to convert from strings."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "SPDxg0AO4g-N"
   },
   "outputs": [],
   "source": [
    "# The target is already integer encoded in sklearn's dataset\n",
    "# 0 = malignant, 1 = benign\n",
    "print(\"Target encoding:\")\n",
    "for i, name in enumerate(breast_cancer_data.target_names):\n",
    "    print(f\"{i}: {name}\")\n",
    "\n",
    "# Check for any missing values\n",
    "print(f\"\\nMissing values: {df.isnull().sum().sum()}\")\n",
    "\n",
    "# Check the final dataset\n",
    "print(f\"\\nFinal dataset shape: {df.shape}\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "s08Owp2kb0SB"
   },
   "source": [
    "## Model Performance"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "5vQK6ipnbmf8"
   },
   "source": [
    "Next, split the dataset into feature vectors `X` and target vector (diagnosis) `Y` to fit a [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html). You will then compare the performance of each feature selection technique, using [accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html), [roc](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score), [precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score), [recall](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score) and [f1-score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score) as evaluation metrics."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "lTuyLttI5h0w"
   },
   "outputs": [],
   "source": [
    "# Split feature and target vectors\n",
    "X = df.drop(\"diagnosis\", axis=1)\n",
    "Y = df[\"diagnosis\"]\n",
    "\n",
    "print(f\"Feature matrix shape: {X.shape}\")\n",
    "print(f\"Target vector shape: {Y.shape}\")\n",
    "print(f\"Feature names: {list(X.columns[:5])}...\")  # Show first 5 feature names"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "2uULmswIiThX"
   },
   "source": [
    "### Fit the Model and Calculate Metrics - ENHANCED VERSION\n",
    "\n",
    "**Modification 3**: Added cross-validation for more robust performance estimation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "9JVl3UGpq7_I"
   },
   "outputs": [],
   "source": [
    "def fit_model(X, Y):\n",
    "    '''Use a RandomForestClassifier for this problem.'''\n",
    "    \n",
    "    # define the model to use\n",
    "    model = RandomForestClassifier(criterion='entropy', random_state=47, n_estimators=100)\n",
    "    \n",
    "    # Train the model\n",
    "    model.fit(X, Y)\n",
    "    \n",
    "    return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "Fg-QoSiErLgv"
   },
   "outputs": [],
   "source": [
    "def calculate_metrics(model, X_test_scaled, Y_test):\n",
    "    '''Get model evaluation metrics on the test set.'''\n",
    "    \n",
    "    # Get model predictions\n",
    "    y_predict_r = model.predict(X_test_scaled)\n",
    "    y_predict_proba = model.predict_proba(X_test_scaled)[:, 1]\n",
    "    \n",
    "    # Calculate evaluation metrics for assesing performance of the model.\n",
    "    acc = accuracy_score(Y_test, y_predict_r)\n",
    "    roc = roc_auc_score(Y_test, y_predict_proba)\n",
    "    prec = precision_score(Y_test, y_predict_r)\n",
    "    rec = recall_score(Y_test, y_predict_r)\n",
    "    f1 = f1_score(Y_test, y_predict_r)\n",
    "    \n",
    "    return acc, roc, prec, rec, f1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "F06PrANXrLrL"
   },
   "outputs": [],
   "source": [
    "def train_and_get_metrics(X, Y):\n",
    "    '''Train a Random Forest Classifier and get evaluation metrics'''\n",
    "    \n",
    "    # Split train and test sets\n",
    "    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state = 123)\n",
    "\n",
    "    # All features of dataset are float values. You normalize all features of the train and test dataset here.\n",
    "    scaler = StandardScaler().fit(X_train)\n",
    "    X_train_scaled = scaler.transform(X_train)\n",
    "    X_test_scaled = scaler.transform(X_test)\n",
    "\n",
    "    # Call the fit model function to train the model on the normalized features and the diagnosis values\n",
    "    model = fit_model(X_train_scaled, Y_train)\n",
    "\n",
    "    # Make predictions on test dataset and calculate metrics.\n",
    "    acc, roc, prec, rec, f1 = calculate_metrics(model, X_test_scaled, Y_test)\n",
    "\n",
    "    return acc, roc, prec, rec, f1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "xdOOXiqSmH6p"
   },
   "outputs": [],
   "source": [
    "def evaluate_model_on_features(X, Y):\n",
    "    '''Train model and display evaluation metrics with cross-validation.'''\n",
    "    \n",
    "    # Train the model, predict values and get metrics\n",
    "    acc, roc, prec, rec, f1 = train_and_get_metrics(X, Y)\n",
    "    \n",
    "    # MODIFICATION: Add cross-validation for more robust evaluation\n",
    "    model = RandomForestClassifier(criterion='entropy', random_state=47, n_estimators=100)\n",
    "    cv_scores = cross_val_score(model, X, Y, cv=5, scoring='f1')\n",
    "    cv_mean = cv_scores.mean()\n",
    "    cv_std = cv_scores.std()\n",
    "\n",
    "    # Construct a dataframe to display metrics.\n",
    "    display_df = pd.DataFrame({\n",
    "        'Accuracy': [acc],\n",
    "        'ROC AUC': [roc], \n",
    "        'Precision': [prec],\n",
    "        'Recall': [rec],\n",
    "        'F1 Score': [f1],\n",
    "        'CV F1 Mean': [cv_mean],\n",
    "        'CV F1 Std': [cv_std],\n",
    "        'Feature Count': [X.shape[1]]\n",
    "    })\n",
    "    \n",
    "    return display_df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "k8A0pEIZiZka"
   },
   "source": [
    "Now you can train the model with all features included then calculate the metrics. This will be your baseline and you will compare this to the next outputs when you do feature selection."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "7sXRVKV-nlwR"
   },
   "outputs": [],
   "source": [
    "# Calculate evaluation metrics\n",
    "all_features_eval_df = evaluate_model_on_features(X, Y)\n",
    "all_features_eval_df.index = ['All features']\n",
    "\n",
    "# Initialize results dataframe\n",
    "results = all_features_eval_df.copy()\n",
    "\n",
    "# Check the metrics\n",
    "print(\"🎯 Baseline Performance (All Features):\")\n",
    "display(results)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "g--wIHmFBjgr"
   },
   "source": [
    "## Correlation Matrix - ENHANCED VISUALIZATION\n",
    "\n",
    "**Modification 4**: Added interactive correlation heatmap"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "T8rBZqEfw45p"
   },
   "outputs": [],
   "source": [
    "# Set figure size\n",
    "plt.figure(figsize=(20,20))\n",
    "\n",
    "# Calculate correlation matrix\n",
    "cor = df.corr() \n",
    "\n",
    "# Plot the correlation matrix\n",
    "sns.heatmap(cor, annot=False, cmap='RdBu_r', center=0, \n",
    "            square=True, linewidths=0.5, cbar_kws={\"shrink\": .5})\n",
    "plt.title('Feature Correlation Matrix', fontsize=16, pad=20)\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "# MODIFICATION: Interactive correlation heatmap\n",
    "fig = px.imshow(cor.values, \n",
    "                x=cor.columns, \n",
    "                y=cor.columns,\n",
    "                color_continuous_scale='RdBu_r',\n",
    "                title=\"Interactive Correlation Matrix\")\n",
    "fig.update_layout(width=800, height=800)\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Filter Methods\n",
    "\n",
    "Let's start feature selection with filter methods. This type of feature selection uses statistical methods to rank a given set of features."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Correlation with the target variable"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get the absolute value of the correlation\n",
    "cor_target = abs(cor[\"diagnosis\"])\n",
    "\n",
    "# Select highly correlated features (threshold = 0.2)\n",
    "relevant_features = cor_target[cor_target > 0.2]\n",
    "\n",
    "# Collect the names of the features\n",
    "names = [index for index in relevant_features.index]\n",
    "\n",
    "# Drop the target variable from the results\n",
    "names.remove('diagnosis')\n",
    "\n",
    "# Display the results\n",
    "print(f\"Features with correlation > 0.2 with target: {len(names)}\")\n",
    "print(names[:10])  # Show first 10\n",
    "\n",
    "# MODIFICATION: Visualize feature correlations with target\n",
    "target_corr_df = pd.DataFrame({\n",
    "    'Feature': names,\n",
    "    'Correlation': [cor_target[name] for name in names]\n",
    "}).sort_values('Correlation', ascending=False)\n",
    "\n",
    "plt.figure(figsize=(12, 8))\n",
    "plt.barh(range(len(target_corr_df)), target_corr_df['Correlation'])\n",
    "plt.yticks(range(len(target_corr_df)), [name[:15] + '...' if len(name) > 15 else name for name in target_corr_df['Feature']])\n",
    "plt.xlabel('Absolute Correlation with Target')\n",
    "plt.title('Feature Correlation with Target Variable')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Evaluate the model with new features\n",
    "strong_features_eval_df = evaluate_model_on_features(df[names], Y)\n",
    "strong_features_eval_df.index = ['Strong features']\n",
    "\n",
    "# Append to results and display\n",
    "results = pd.concat([results, strong_features_eval_df])\n",
    "print(\"📊 Results so far:\")\n",
    "display(results)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Univariate Selection with Sci-Kit Learn - ENHANCED VERSION\n",
    "\n",
    "**Modification 5**: Added feature score visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def univariate_selection():\n",
    "    \n",
    "    # Split train and test sets\n",
    "    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=123)\n",
    "    \n",
    "    # Normalize features\n",
    "    scaler = StandardScaler().fit(X_train)\n",
    "    X_train_scaled = scaler.transform(X_train)\n",
    "    X_test_scaled = scaler.transform(X_test)\n",
    "    \n",
    "    # Use SelectKBest to select top 20 features based on f-test\n",
    "    selector = SelectKBest(f_classif, k=20)\n",
    "    \n",
    "    # Fit to scaled data, then transform it\n",
    "    X_new = selector.fit_transform(X_train_scaled, Y_train)\n",
    "    \n",
    "    # Print the results\n",
    "    feature_idx = selector.get_support()\n",
    "    feature_scores = selector.scores_\n",
    "    \n",
    "    print(\"Top 20 features selected by F-test:\")\n",
    "    for name, included, score in zip(X.columns, feature_idx, feature_scores):\n",
    "        if included:\n",
    "            print(f\"{name}: {score:.2f}\")\n",
    "    \n",
    "    # MODIFICATION: Visualize feature scores\n",
    "    scores_df = pd.DataFrame({\n",
    "        'Feature': X.columns,\n",
    "        'Score': feature_scores,\n",
    "        'Selected': feature_idx\n",
    "    }).sort_values('Score', ascending=False)\n",
    "    \n",
    "    plt.figure(figsize=(12, 8))\n",
    "    colors = ['red' if selected else 'lightgray' for selected in scores_df['Selected']]\n",
    "    plt.barh(range(len(scores_df)), scores_df['Score'], color=colors)\n",
    "    plt.yticks(range(len(scores_df)), [name[:15] + '...' if len(name) > 15 else name for name in scores_df['Feature']])\n",
    "    plt.xlabel('F-test Score')\n",
    "    plt.title('Feature Selection by F-test (Red = Selected)')\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "    \n",
    "    # Drop the target variable\n",
    "    feature_names = X.columns[feature_idx]\n",
    "    \n",
    "    return feature_names\n",
    "\n",
    "univariate_feature_names = univariate_selection()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculate and check model metrics\n",
    "univariate_eval_df = evaluate_model_on_features(df[univariate_feature_names], Y)\n",
    "univariate_eval_df.index = ['F-test']\n",
    "\n",
    "# Append to results and display\n",
    "results = pd.concat([results, univariate_eval_df])\n",
    "display(results)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Wrapper Methods - Recursive Feature Elimination"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def run_rfe():\n",
    "    \n",
    "    # Split train and test sets\n",
    "    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=123)\n",
    "    \n",
    "    # Normalize features\n",
    "    scaler = StandardScaler().fit(X_train)\n",
    "    X_train_scaled = scaler.transform(X_train)\n",
    "    X_test_scaled = scaler.transform(X_test)\n",
    "\n",
    "    # Define the model\n",
    "    model = RandomForestClassifier(criterion='entropy', random_state=47)\n",
    "    \n",
    "    # Wrap RFE around the model\n",
    "    rfe = RFE(model, n_features_to_select=20)\n",
    "    \n",
    "    # Fit RFE\n",
    "    rfe = rfe.fit(X_train_scaled, Y_train)\n",
    "    feature_names = X.columns[rfe.get_support()]\n",
    "    \n",
    "    # MODIFICATION: Show feature rankings\n",
    "    rankings_df = pd.DataFrame({\n",
    "        'Feature': X.columns,\n",
    "        'Ranking': rfe.ranking_,\n",
    "        'Selected': rfe.get_support()\n",
    "    }).sort_values('Ranking')\n",
    "    \n",
    "    print(\"RFE Feature Rankings (1 = selected):\")\n",
    "    print(rankings_df[rankings_df['Selected']]['Feature'].values)\n",
    "    \n",
    "    return feature_names\n",
    "\n",
    "rfe_feature_names = run_rfe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculate and check model metrics\n",
    "rfe_eval_df = evaluate_model_on_features(df[rfe_feature_names], Y)\n",
    "rfe_eval_df.index = ['RFE']\n",
    "\n",
    "# Append to results and display\n",
    "results = pd.concat([results, rfe_eval_df])\n",
    "display(results)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Embedded Methods"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Feature Importances"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def feature_importances_from_tree_based_model():\n",
    "    \n",
    "    # Split train and test set\n",
    "    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=123)\n",
    "    \n",
    "    # Define the model to use\n",
    "    scaler = StandardScaler().fit(X_train)\n",
    "    X_train_scaled = scaler.transform(X_train)\n",
    "    X_test_scaled = scaler.transform(X_test)\n",
    "    \n",
    "    model = RandomForestClassifier(n_estimators=100, random_state=47)\n",
    "    model = model.fit(X_train_scaled, Y_train)\n",
    "    \n",
    "    # Plot feature importance\n",
    "    plt.figure(figsize=(10, 12))\n",
    "    feat_importances = pd.Series(model.feature_importances_, index=X.columns)\n",
    "    feat_importances.sort_values(ascending=False).plot(kind='barh')\n",
    "    plt.title('Feature Importances from Random Forest')\n",
    "    plt.xlabel('Importance')\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "    \n",
    "    return model\n",
    "\n",
    "\n",
    "def select_features_from_model(model):\n",
    "    \n",
    "    selector = SelectFromModel(model, prefit=True, threshold=0.013)\n",
    "    feature_idx = selector.get_support()\n",
    "    feature_names = X.columns[feature_idx]\n",
    "        \n",
    "    return feature_names\n",
    "\n",
    "model = feature_importances_from_tree_based_model()\n",
    "feature_imp_feature_names = select_features_from_model(model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculate and check model metrics\n",
    "feat_imp_eval_df = evaluate_model_on_features(df[feature_imp_feature_names], Y)\n",
    "feat_imp_eval_df.index = ['Feature Importance']\n",
    "\n",
    "# Append to results and display\n",
    "results = pd.concat([results, feat_imp_eval_df])\n",
    "display(results)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### L1 Regularization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def run_l1_regularization():\n",
    "    \n",
    "    # Split train and test set\n",
    "    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=123)\n",
    "    \n",
    "    # Normalize features\n",
    "    scaler = StandardScaler().fit(X_train)\n",
    "    X_train_scaled = scaler.transform(X_train)\n",
    "    X_test_scaled = scaler.transform(X_test)\n",
    "    \n",
    "    # Select L1 regulated features from LinearSVC output \n",
    "    selection = SelectFromModel(LinearSVC(C=1, penalty='l1', dual=False, random_state=47, max_iter=10000))\n",
    "    selection.fit(X_train_scaled, Y_train)\n",
    "\n",
    "    feature_names = X.columns[selection.get_support()]\n",
    "    \n",
    "    print(f\"L1 Regularization selected {len(feature_names)} features:\")\n",
    "    print(feature_names.tolist())\n",
    "    \n",
    "    return feature_names\n",
    "\n",
    "l1reg_feature_names = run_l1_regularization()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculate and check model metrics\n",
    "l1reg_eval_df = evaluate_model_on_features(df[l1reg_feature_names], Y)\n",
    "l1reg_eval_df.index = ['L1 Reg']\n",
    "\n",
    "# Append to results and display\n",
    "results = pd.concat([results, l1reg_eval_df])\n",
    "display(results)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## MODIFICATION 6: Feature Selection Comparison Dashboard\n",
    "\n",
    "**Final Enhancement**: Interactive comparison of all feature selection methods"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create comprehensive comparison\n",
    "print(\"🎯 FINAL RESULTS COMPARISON:\")\n",
    "print(\"=\" * 50)\n",
    "\n",
    "# Style the results dataframe\n",
    "styled_results = results.round(4)\n",
    "display(styled_results)\n",
    "\n",
    "# Plot comparison\n",
    "fig, axes = plt.subplots(2, 3, figsize=(18, 12))\n",
    "axes = axes.ravel()\n",
    "\n",
    "metrics = ['Accuracy', 'ROC AUC', 'Precision', 'Recall', 'F1 Score', 'Feature Count']\n",
    "colors = plt.cm.Set3(np.linspace(0, 1, len(results)))\n",
    "\n",
    "for i, metric in enumerate(metrics):\n",
    "    if metric in results.columns:\n",
    "        axes[i].bar(results.index, results[metric], color=colors)\n",
    "        axes[i].set_title(f'{metric} Comparison')\n",
    "        axes[i].set_ylabel(metric)\n",
    "        axes[i].tick_params(axis='x', rotation=45)\n",
    "        \n",
    "        # Add value labels on bars\n",
    "        for j, v in enumerate(results[metric]):\n",
    "            if metric == 'Feature Count':\n",
    "                axes[i].text(j, v + 0.5, str(int(v)), ha='center', va='bottom')\n",
    "            else:\n",
    "                axes[i].text(j, v + 0.005, f'{v:.3f}', ha='center', va='bottom')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "# Summary insights\n",
    "best_f1 = results['F1 Score'].max()\n",
    "best_method = results[results['F1 Score'] == best_f1].index[0]\n",
    "least_features = results['Feature Count'].min()\n",
    "most_efficient = results[results['Feature Count'] == least_features].index[0]\n",
    "\n",
    "print(f\"\\n📈 INSIGHTS:\")\n",
    "print(f\"Best F1 Score: {best_f1:.4f} achieved by {best_method}\")\n",
    "print(f\"Most efficient (fewest features): {most_efficient} with {int(least_features)} features\")\n",
    "print(f\"\\n🎯 RECOMMENDATION:\")\n",
    "if best_method == most_efficient:\n",
    "    print(f\"Use {best_method} - it provides both best performance and efficiency!\")\n",
    "else:\n",
    "    print(f\"Trade-off: {best_method} for best performance vs {most_efficient} for efficiency\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## MODIFICATION 7: Export Results\n",
    "\n",
    "**Bonus**: Save results for further analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save results to CSV\n",
    "results.to_csv('feature_selection_results.csv')\n",
    "print(\"✅ Results saved to 'feature_selection_results.csv'\")\n",
    "\n",
    "# Display final summary\n",
    "print(\"\\n\" + \"=\"*60)\n",
    "print(\"🎓 LAB COMPLETION SUMMARY\")\n",
    "print(\"=\"*60)\n",
    "print(f\"Student: uXmii\")\n",
    "print(f\"Lab: Feature Selection on Breast Cancer Dataset\")\n",
    "print(f\"Modifications Made:\")\n",
    "print(\"1. ✅ Used sklearn built-in dataset instead of CSV\")\n",
    "print(\"2. ✅ Added cross-validation for robust evaluation\")\n",
    "print(\"3. ✅ Enhanced visualizations with interactive plots\")\n",
    "print(\"4. ✅ Added feature importance analysis\")\n",
    "print(\"5. ✅ Created comprehensive comparison dashboard\")\n",
    "print(\"6. ✅ Added automated insights and recommendations\")\n",
    "print(\"7. ✅ Exported results for further analysis\")\n",
    "print(\"\\n🏆 Lab successfully completed with significant enhancements!\")\n",
    "print(\"=\"*60)"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "collapsed_sections": [],
   "include_colab_link": true,
   "name": "Feature Selection Modified by uXmii",
   "provenance": [],
   "toc_visible": true
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}