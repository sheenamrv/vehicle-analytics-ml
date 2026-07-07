from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.model.model_utils import get_rf_config

def build_model(model_type, parameters):

    if model_type == "svm":
        return SVC(
            C=parameters.get("C", 1.0),
            probability=True,
            random_state=42
        )

    elif model_type == "knn":
        return KNeighborsClassifier(n_neighbors=parameters.get("n_neighbors",5))

    elif model_type == "decision_tree":
        return DecisionTreeClassifier(max_depth=parameters.get("max_depth"))

    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=parameters.get("n_estimators",100),
            max_depth=parameters.get("max_depth"),
            random_state=42
        )

    elif model_type == "logistic_regression":
        return LogisticRegression(max_iter=1000)

    return None

def get_model_parameters(model_type):
    
    if model_type == "random_forest":
        return get_rf_config()
    elif model_type == "knn":
        
        neighbours = input("Number of neighbors [5]: ").strip()
        
        return {
            "n_neighbors" : int(neighbours) if neighbours else 5
        }
        
    elif model_type == "svm":
        
        c = input("C value [1.0]: ").strip()
        
        return {
            "C" : float(c) if c else 1.0
        }
    
    elif model_type == "decision_tree":
        
        depth = input("Max depth [None]: ").strip()
        
        return {
            "max_depth" : int(depth) if depth else None
        }
        
    return {}


    