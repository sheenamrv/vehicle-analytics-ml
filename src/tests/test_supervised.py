import pandas as pd

from src.model.supervised_model import get_model_parameters, run_supervised_workflow


def generate_df():
    return pd.DataFrame({
        "feature1": [0, 0, 1, 1, 0, 1, 0, 1],
        "feature2": [0, 1, 0, 1, 0, 1, 1, 0],
        "category": ["a", "a", "b", "b", "a", "b", "a", "b"],
        "label": [0, 0, 1, 1, 0, 1, 0, 1],
    })


def test_get_model_parameters_uses_passed_parameters_without_prompting():
    params = {"n_estimators": 10, "max_depth": 2}

    assert get_model_parameters("random_forest", parameters=params) == params


def test_run_supervised_workflow_returns_frontend_friendly_result():
    df = generate_df()

    result = run_supervised_workflow(
        df=df,
        label_col="label",
        model_type="decision_tree",
        parameters={"max_depth": 2},
        test_size=0.25,
        random_state=42,
        stratify=False,
    )

    assert result["model_type"] == "decision_tree"
    assert result["parameters"] == {"max_depth": 2}
    assert "model" in result
    assert "features" in result
    assert "metrics" in result
    assert "confusion_matrix" in result
    assert "predictions" in result
    assert "X_test" in result
    assert "y_test" in result
