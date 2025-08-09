import pytest
from utils import generate_embedding


def test_single_string_embedding():
    text = "Hello world"
    emb = generate_embedding([text])
    assert isinstance(emb, list)
    assert all(isinstance(val, float) for val in emb), "Embedding must be list of floats"
    assert len(emb) > 0, "Embedding should not be empty"


def test_multiple_strings_embedding():
    texts = ["Hello", "World"]
    embeddings = [generate_embedding([text]) for text in texts]
    for emb in embeddings:
        assert isinstance(emb, list)
        assert all(isinstance(val, float) for val in emb)


def test_empty_string_error():
    with pytest.raises(ValueError):
        generate_embedding([""])
