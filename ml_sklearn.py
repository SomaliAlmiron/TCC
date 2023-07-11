import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
import argparse
from sklearn.model_selection import train_test_split
import pandas as pd
from urllib.parse import urlparse
from imblearn.combine import SMOTEENN
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from mlflow.tracking import MlflowClient
import os 

parser = argparse.ArgumentParser()
parser.add_argument("inputPath", help="arquivo input Tcc", type=str)
parser.add_argument("experiment", help="experimento MLFLOW", type=str)
parser.add_argument("modelName", help="nome modelo MLfLow", type=str)
parser.add_argument("n_estimators", help="parametro n_estimators", default=100,type=int)
parser.add_argument("max_depth", help="parametro max_depth", default=2,type=int)
parser.add_argument("random_state", help="parametro random_state", default=0, type=int)
args = parser.parse_args()


df = pd.read_csv(args.inputPath, sep=",")

X = df.drop(["autismo"], axis = 1)    
Y = df['autismo']

smoteenn = SMOTEENN()
X_smoteenn, Y_smoteenn = smoteenn.fit_resample(X, Y)

X_treino_smoteenn, X_teste_smoteenn, Y_treino_smoteenn, Y_teste_smoteenn = train_test_split(X_smoteenn, Y_smoteenn, test_size = 0.25, stratify = Y_smoteenn, random_state=0) #random state é o que nos garantirá que sempre que rodarmos o código, o resultado será o mesmo.

try:
    idExperiment = mlflow.create_experiment(args.experiment)
except:
    idExperiment = mlflow.get_experiment_by_name(args.experiment).experiment_id

with mlflow.start_run(experiment_id=idExperiment):
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("random_state", args.random_state)

    model_rand_ab = AdaBoostClassifier(RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=args.random_state))
    
    cv = KFold(n_splits = 5, shuffle = True)
    results_rand_ab_smoteen = cross_val_predict(model_rand_ab, X_treino_smoteenn, Y_treino_smoteenn, cv = cv)

    mlflow.log_metric("accuracy_score", accuracy_score(Y_treino_smoteenn.values,results_rand_ab_smoteen.round()))


    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    if tracking_url_type_store != "file":
        mlflow.sklearn.log_model(model_rand_ab, "model", registered_model_name=args.modelName)
    else:
        mlflow.sklearn.log_model(model_rand_ab, "model")

