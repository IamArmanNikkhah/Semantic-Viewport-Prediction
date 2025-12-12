from dataclasses import dataclass
from common.interfaces import Ms, TileId 
# Data class to hold detection information
@dataclass
class Detection:
    timestamp: Ms
    identified_semantic_class: str
    confidence: float
    tile_id: TileId
