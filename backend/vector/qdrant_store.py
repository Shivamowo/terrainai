# vector/qdrant_store.py

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType, PointStruct
import uuid

from qdrant_client.models import Filter, FieldCondition, MatchValue



COLLECTION_NAME = "terrain_patches"
EMBEDDING_DIM = 512


class TerrainVectorStore:
    def __init__(self):
        # 👇 THIS replaces Docker completely
        self.client = QdrantClient(path="./qdrant_data")  # in-process Qdrant

        self._create_collection()
        self._create_indexes()

    def _create_collection(self):
        if COLLECTION_NAME not in [c.name for c in self.client.get_collections().collections]:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE
                )
            )

    def _create_indexes(self):
        self.client.create_payload_index(
            COLLECTION_NAME, "class_id", PayloadSchemaType.INTEGER
        )
        self.client.create_payload_index(
            COLLECTION_NAME, "is_rare_class", PayloadSchemaType.BOOL
        )
        self.client.create_payload_index(
            COLLECTION_NAME, "iou_score", PayloadSchemaType.FLOAT
        )

    # This is what trainer.py will call after each epoch
    def index_batch(self, batch_points: list[dict]):
        points = []

        for item in batch_points:
            points.append(
                PointStruct(
                id=str(uuid.uuid4()),
                vector=item["vector"],
                payload={
                    "image_path": item["image_path"],
                    "crop_bbox": item.get("crop_bbox"),
                    "class_id": item["class_id"],
                    "class_name": item["class_name"],
                    "iou": item["iou"],
                    "epoch": item["epoch"],
                    "split": item["split"],
                    "run_id": item["run_id"],
                    "is_rare_class": item["is_rare_class"],
                    "image_width": item["image_width"],
                    "image_height": item["image_height"],
                },
            )
        )

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        

    # This is what FastAPI will call for Failure Browser
   
    def get_hard_examples(self, class_id: int, limit: int = 16):
        results = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=[1.0] * EMBEDDING_DIM,
            limit=limit,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="class_id",
                        match=MatchValue(value=class_id)
                    )
                ]
            ),
            with_payload=True
        )

        results = sorted(results, key=lambda r: r.payload["iou"])

        return [
            {
                "image_path": r.payload["image_path"],
                "iou": r.payload["iou"],
                "epoch": r.payload["epoch"],
            }
            for r in results
        ]