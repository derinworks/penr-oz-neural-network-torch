import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from main import app


client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def mock_new_model():
    with patch("main.NeuralNetworkModel") as MockModel:
        mock_instance = MagicMock()
        MockModel.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_deserialized_model():
    with patch("neural_net_model.NeuralNetworkModel.deserialize") as mock_deserialize:
        mock_instance = MagicMock()
        mock_deserialize.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_delete_model():
    with patch("neural_net_model.NeuralNetworkModel.delete") as mock_delete:
        yield mock_delete


def test_redirect_to_docs():
    response = client.get("/")
    assert response.status_code == 200
    assert response.url.path == "/docs"


def test_create_model_endpoint(mock_new_model):
    payload = {
        "model_id": "test",
        "layer_sizes": [9, 9, 9],
        "weight_algo": "xavier",
        "bias_algo": "random",
        "activation_algos": ["sigmoid"] * 2,
    }

    response = client.post("/model/", json=payload)

    assert response.status_code == 200

    assert response.json() == {
        "message": "Model test created and saved successfully"
    }

    mock_new_model.serialize.assert_called_once()


def test_output_endpoint(mock_deserialized_model):
    mock_deserialized_model.compute_output.return_value = ([0, 1, 0], None)

    payload = {
        "model_id": "test",
        "input": {
            "activation_vector": [0, 0, 0]
        }
    }

    response = client.post("/output/", json=payload)

    assert response.json() == {
        "output_vector": [0, 1, 0],
        "cost": None,
    }

    assert response.status_code == 200


def test_train_endpoint(mock_deserialized_model):
    payload = {
        "model_id": "test",
        "training_data": [
            {
                "activation_vector": [0, 0, 0],
                "target_vector":     [0, 1, 0]
            }
        ],
        "epochs": 2,
        "learning_rate": .01
    }

    response = client.put("/train/", json=payload)

    assert response.status_code == 202

    assert response.json() == {
        "message": "Training started asynchronously."
    }

    mock_deserialized_model.train.assert_called_once()


def test_progress_endpoint(mock_deserialized_model):
    mock_deserialized_model.progress = [
        "Some progress"
    ]
    mock_deserialized_model.avg_cost = 0.123

    response = client.get("/progress/", params={"model_id": "test"})

    assert response.status_code == 200

    assert response.json() == {
        "progress": [
            "Some progress"
        ],
        "average_cost": 0.123
    }

def test_stats_endpoint(mock_deserialized_model):
    mock_deserialized_model.stats = {
        "some": "stats",
    }

    response = client.get("/stats/", params={"model_id": "test"})

    assert response.status_code == 200

    assert response.json() == {
        "some": "stats",
    }

def test_not_found(mock_deserialized_model):
    mock_deserialized_model.compute_output.side_effect = KeyError("Testing key error :-)")

    response = client.post("/output/", json={
        "model_id": "nonexistent",
        "input": {
            "activation_vector": [0, 0, 0]
        }
    })

    assert response.status_code == 404
    assert response.json() == {'detail': "Not found error occurred: 'Testing key error :-)'"}


def test_invalid_payload():
    response = client.post("/output/", json={
        "model_id": "test",
        # Missing "input" key
    })

    assert response.status_code == 422
    assert "detail" in response.json()


def test_value_error(mock_deserialized_model):
    mock_deserialized_model.compute_output.side_effect = ValueError("Testing value error :-)")

    response = client.post("/output/", json={
        "model_id": "test",
        "input": {
            "activation_vector": [0, 0, 0]
        }
    })

    assert response.status_code == 400
    assert response.json() == {'detail': 'Value error occurred: Testing value error :-)'}


def test_unhandled_exception(mock_deserialized_model):
    mock_deserialized_model.compute_output.side_effect = RuntimeError("Unexpected error")

    response = client.post("/output/", json={
        "model_id": "test",
        "input": {
            "activation_vector": [0, 0, 0]
        }
    })

    assert response.status_code == 500
    assert response.json() == {"detail": "Please refer to server logs"}


def test_delete_model_endpoint(mock_delete_model):
    response = client.delete("/model/", params={"model_id": "test"})

    assert response.status_code == 204

    mock_delete_model.assert_called_once()
