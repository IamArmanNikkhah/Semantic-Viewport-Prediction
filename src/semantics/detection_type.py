from dataclasses import dataclass

# Data class to hold detection information
@dataclass
class Detection:
    timestamp: float
    identified_semantic_class: str
    confidence: float
    tile_id: int
