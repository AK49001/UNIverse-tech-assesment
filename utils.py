import boto3, json, io
import os
import numpy as np
from typing import List

# ENV VARIABLES:
AWS_TYPE = "us-east-1"
S3_BUCKET = "rag-vector-bucket"

# Instantiating bedrock client and s3 bucket client


bedrock = boto3.client("bedrock-runtime", region_name=AWS_TYPE)
s3 = boto3.client("s3", region_name=AWS_TYPE)


def generate_embedding(texts: List[str]) -> List[List[float]]:

    embeddings = []
    for text in texts:
        payload = {"inputText": text}
        response = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload)
        )
        data = json.loads(response["body"].read())
        emb = data.get("embedding") or data.get("embeddings") or data.get("output", {}).get("embedding")
        if emb is None:
            raise RuntimeError(f"Invalid response: {data}")
        else:
            embeddings.append(emb)
    return embeddings


def compute_cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0
    cosine_sim = dot_product / (norm_a * norm_b)
    return cosine_sim


def upload_to_s3(key: str, data: dict):
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=json.dumps(data))


def get_s3_embeddings():
    s3_objects = s3.list_objects_v2(Bucket=S3_BUCKET).get("Contents", [])
    info = []
    for object in s3_objects:
        file_data = s3.get_object(Bucket=S3_BUCKET, Key=object["Key"])["Body"].read()
        info.append(json.loads(file_data))
    return info
