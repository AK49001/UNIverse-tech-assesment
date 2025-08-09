import pytest
from utils import generate_embedding
from unittest.mock import MagicMock


def test_generate_embeddings_success():

    mock_client = MagicMock()
    mock_model_id = "fake-model-id"
    mock_texts = ["Hello world", "Bedrock test"]
    mock_response = {"embedding": [[0.1, 0.2], [0.3, 0.4]]}

    mock_client.invoke_model.return_value = mock_response
    result = generate_embedding(mock_client)
    mock_client.invoke_model.assert_called_once_with(
        modelId=mock_model_id,
        body={"inputText": mock_texts}
    )
    assert result == mock_response


def test_generate_embeddings_invalid_input():
    mock_client = MagicMock()

    with pytest.raises(ValueError):
        generate_embedding(mock_client)

    assert "Input texts must be a list"


if __name__ == '__main__':
    unittest.main()

