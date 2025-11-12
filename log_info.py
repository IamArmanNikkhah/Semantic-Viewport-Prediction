from dataclasses import dataclass

# Data classes to hold information about log files and frame data
@dataclass
class FrameData:
    pitch: int
    roll: float
    sec: float
    yaw: float

@dataclass
class LogInfo:
    filename: str
    hmd: str
    data: list[FrameData]  # list of dataclass with pitch/roll/sec/yaw
    log_type: str  # "json", "csv", "text"
    label: str = None 
    video_length_s: float = None