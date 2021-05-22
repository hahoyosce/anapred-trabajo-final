import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
from sklearn.preprocessing import MinMaxScaler
from operator import itemgetter
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


def corr_matrix_heatmap(cor):
    plt.figure(figsize=(math.floor(len(cor.columns) + 2), math.floor(len(cor.columns))))
    sns.heatmap(cor, annot=True, cmap=plt.cm.Blues)
    plt.show()


def quantitative_var_corr(df, cor):
    cor_tril = pd.DataFrame(np.tril(cor), index=[row for row in cor], columns=[row for row in cor])

    cor_list = []
    for row in df._get_numeric_data().columns:
        for col in df._get_numeric_data().columns:
            if cor_tril[row][col] != 1 and cor_tril[row][col] != 0:
                cor_list.append([abs(cor_tril[row][col]), row, col])
    cor_list = sorted(cor_list, key=itemgetter(0), reverse=True)

    plt.figure(figsize=(math.floor((len(cor_tril.columns) ** 2 - len(cor_tril.columns)) / 2 * 6), 4))
    for i, pair in enumerate(cor_list, 1):
        axs = plt.subplot(1, len(cor_list), i)
        axs.set_title('Cor = ' + str(pair[0]))
        sns.scatterplot(data=df, x=pair[1], y=pair[2])


def quantitative_var_distplot(df):
    skew_list = []
    for column in df._get_numeric_data().columns:
        skew_list.append([abs(df.skew()[column]), column])
    skew_list = sorted(skew_list, key=itemgetter(0), reverse=True)

    plt.figure(figsize=(math.floor(len(skew_list) * 6), 4))
    for i, col in enumerate(skew_list, 1):
        axs = plt.subplot(1, len(skew_list), i)
        axs.set_title('Skew = ' + str(col[0]))
        sns.histplot(data=df[col[1]], kde=True)


def categorical_var_histplot(df):
    # Obtener los gráficos de ocurrencias de variables categóricas siempre y cuando tengan menos de 100 valores (para evitar graficas valores de datetime, por ejemplo).
    categorical = [column for column in df.columns if column not in df._get_numeric_data().columns and df[column].nunique() < 100]

    if categorical:
        plt.figure(figsize=(math.floor(len(categorical) * 6), 4))
        for i, col in enumerate(categorical, 1):
            axs = plt.subplot(1, len(categorical), i)
            sns.histplot(data=df, y=col)


def preprocessing(df, depVar):

    # Borrar los registros con alguna variable no numérica, si corresponde (NaN).
    df = df.copy().dropna()

    # Convertir todas las variables string correspondientes a fechas a datetime.
    dt_mask = df.astype(str).apply(lambda x: x.str.match(r'(\d{2,4}(\/|-)\d{2}(\/|-)\d{2,4})+').all())
    df.loc[:, dt_mask] = df.loc[:, dt_mask].apply(pd.to_datetime)

    # Obtener año, mes, día y hora de estas variables datetime.
    for column in df.select_dtypes(include=['datetime64[ns]']):
        df[column+'_year'] = pd.DatetimeIndex(df[column]).year
        df[column+'_month'] = pd.DatetimeIndex(df[column]).month
        df[column+'_day'] = pd.DatetimeIndex(df[column]).day
        df[column+'_hour'] = pd.DatetimeIndex(df[column]).hour

        # Eliminar la columna de formato datetime.
        del df[column]
    
    # Seleccionar la variable dependiente en la regresión.
    y = df[depVar]
    del df[depVar]

    # Codificar en dummies todas las variables todas las variables categóricas.
    categorical = [column for column in df.columns if column not in df._get_numeric_data().columns and df[column].nunique() > 100]
    for column in categorical:
        del df[column]
    df = pd.get_dummies(df)
    
    # Se normalizan los valores a valores entre 0 y 1.
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(df),columns=df.columns)

    return X, y


def grid_cv(X_train, y_train, regressor, parameters, cv=5, scoring=None):

    grid = GridSearchCV(
        regressor,
        parameters,
        cv=cv,
        scoring=scoring
    )

    grid.fit(X_train, y_train)

    return grid


def best_regressors(regressors, models):
    best_regressors = []
    for regressor in regressors:
        best_regressors.append([models[regressor].best_score_, str(models[regressor].best_estimator_).replace('\n', ''), str(models[regressor].best_params_)])
    best_regressors = pd.DataFrame(best_regressors, index=list(models.keys()), columns=['Best score', 'Best estimator', 'Best parameters'])

    return best_regressors


def predict_real_scatterplot(regressors, models, X_test, y_test):
    predicts = {}

    plt.figure(figsize=(math.floor(len(regressors) * 7), 4))
    for i, regressor in enumerate(regressors, 1):
        axs = plt.subplot(1, len(regressors), i)
        predicts[regressor] = models[regressor].predict(X_test)
        sns.scatterplot(x=predicts[regressor], y=y_test)
        sns.lineplot(x=predicts[regressor], y=predicts[regressor], linestyle='--', color='k')
        plt.xlabel("Predicción")
        plt.ylabel("Real")
        plt.title("Modelo: " + list(models)[i-1])

    return predicts

def predict_residuals_scatterplot(regressors, models, predicts, y_test):
    plt.figure(figsize=(math.floor(len(regressors) * 7), 4))
    for i, regressor in enumerate(regressors, 1):
        axs = plt.subplot(1, len(regressors), i)
        sns.scatterplot(x=predicts[regressor], y=y_test - predicts[regressor])
        sns.lineplot(x=predicts[regressor], y=np.zeros(len(predicts[regressor])), linestyle='--', color='k')
        plt.xlabel("Predicción")
        plt.ylabel("Residuales")
        plt.title("Modelo: " + list(models)[i-1])

def eval_metrics(regressors, predicts, y_test):

    eval_metrics = {
        'explained_variance': metrics.explained_variance_score,
        'max_error': metrics.max_error,
        'mean_absolute_error': metrics.mean_absolute_error,
        'mean_squared_error': metrics.mean_squared_error,
        #'mean_squared_log_error': metrics.mean_squared_log_error,
        'median_absolute_error': metrics.median_absolute_error,
        #'mean_poisson_deviance': metrics.mean_poisson_deviance,
        #'mean_gamma_deviance': metrics.mean_gamma_deviance,
        #'mean_tweedie_deviance': metrics.mean_tweedie_deviance,
        'r2_score': metrics.r2_score,
    }

    model_metrics = []
    for i, regressor in enumerate(regressors):
        model_metrics.append([])
        for metric in eval_metrics:
            model_metrics[i].append(eval_metrics[metric](y_test, predicts[regressor]))
    model_metrics = pd.DataFrame(model_metrics, index=list(predicts.keys()), columns=list(eval_metrics.keys()))

    return model_metrics