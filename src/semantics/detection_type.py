from dataclasses import dataclass

SEMANTIC_CLASSES: list[str] = [
    "faces",
    "bodies",
    "vehicles",
    "animals",
    "text",
    "horizon/skyline",
    "moving_objects",
    "active_speaker",
]

@dataclass
class Detection:
    timestamp: float
    identified_semantic_class: str
    confidence: float
    tile_id: int
