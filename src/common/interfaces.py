# interfaces.py
# =============================================================================
# API Contract for the 360° Streaming Prototype
# -----------------------------------------------------------------------------
# Conventions (MUST be followed across the codebase):
# - Time units: integer milliseconds (Ms).
# - Angles: radians in [-π, π] for yaw, and [-π/2, π/2] for pitch.
# - ERP tiling: fixed 6 (rows) x 12 (cols) grid, TileId ∈ [0, 71].
# - Semantic maps: shape [K][TILE_ROWS][TILE_COLS], normalized to [0, 1].
# - Decision latency budget: ≤ 20 ms for on-device control path per chunk.
# - Equal average bandwidth is enforced across methods in experiments.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypedDict,
    Literal,
    NewType,
    runtime_checkable,
)

# ---------- Constants / global invariants ----------
TILE_ROWS: int = 4
TILE_COLS: int = 6
NUM_TILES: int = TILE_ROWS * TILE_COLS  # 24

# Default taxonomy (can be swapped, but keep K and order consistent at runtime)
'''
DEFAULT_CLASSES: Tuple[str, ...] = (
    "face",
    "body",
    "vehicle",
    "animal",
    "text",
    "horizon_skyline",
    "moving_object",
    "active_speaker",
)
'''
DEFAULT_CLASSES: Tuple[str, ...] = (
    "human face",
    "person",
    "vehicle",
    "animal",
    "text logo",
    "sports ball",
    "fireworks",
    "waterfall",
    "toy gun",
    "mountain",
    "building",
)

# ---------- Strongly-typed primitives ----------
Ms = NewType("Ms", int)                 # integer milliseconds
Radians = NewType("Radians", float)     # angle in radians
Kbps = NewType("Kbps", int)             # kilobits per second

TileId = NewType("TileId", int)         # 0..NUM_TILES-1
BitrateLevel = NewType("BitrateLevel", int)  # discrete level index (e.g., 0..4)

# ---------- Data carried through the control path ----------

@dataclass(frozen=True)
class HeadMotionSample:
    """Single head-pose observation at time t_ms."""
    t_ms: Ms
    yaw_rad: Radians      # [-π, π]
    pitch_rad: Radians    # [-π/2, π/2]
    ang_vel_rad_s: float  # scalar angular speed (radians/second)


@dataclass(frozen=True)
class HeadMotionHistory:
    """
    Sliding window of recent head motion samples.

    Invariants:
    - len(samples) ≥ 2
    - t_ms is strictly increasing
    - All angles obey the global conventions (radians; bounded ranges).
    """
    samples: Sequence[HeadMotionSample]

    def t_start(self) -> Ms:
        return self.samples[0].t_ms

    def t_end(self) -> Ms:
        return self.samples[-1].t_ms

    def duration_ms(self) -> Ms:
        return Ms(int(self.t_end()) - int(self.t_start()))


@dataclass(frozen=True)
class SemanticMap:
    """
    K-class tile weight map at (or for) time t_ms.

    Shape:
    - weights: [K][TILE_ROWS][TILE_COLS], values in [0, 1].
    Contract:
    - len(classes) == len(weights)
    - classes order is consistent across the run
    - Normalization (per class and/or global) is handled upstream.
    """
    t_ms: Ms
    classes: Tuple[str, ...]
    weights: Sequence[Sequence[Sequence[float]]]  # K x 6 x 12


@dataclass(frozen=True)
class PersonalizationWeights:
    """
    Per-user class weights (kept on-device).
    Contract:
    - len(classes) == len(w)
    - classes order matches SemanticMap.classes for fusion.
    """
    classes: Tuple[str, ...]
    w: Tuple[float, ...]  # one weight per class


@dataclass(frozen=True)
class FusedSemanticMap:
    """
    Result of applying PersonalizationWeights to a SemanticMap.
    Typically: softmax(w ⊙ P_sem) or similar normalization.
    """
    t_ms: Ms
    classes: Tuple[str, ...]
    weights: Sequence[Sequence[Sequence[float]]]  # K x 6 x 12 (personalized)


@dataclass(frozen=True)
class PredictedFoV:
    """
    Point prediction + uncertainty for a given horizon.
    - horizon_ms: e.g., 1000..1500 for prefetch; 200..300 for enhance.
    - entropy: predictor uncertainty (e.g., via MC-dropout); non-negative.
    """
    horizon_ms: Ms
    yaw_rad: Radians
    pitch_rad: Radians
    entropy: float


@dataclass(frozen=True)
class FoVTileDensity:
    """
    Optional tile-level FoV density (probabilities).
    Contract:
    - probs has shape [TILE_ROWS][TILE_COLS]
    - sum of all probs ∈ [0.99, 1.01] (tolerant to rounding)
    """
    horizon_ms: Ms
    probs: Sequence[Sequence[float]]


@dataclass(frozen=True)
class BanditAction:
    """
    Discrete control knobs selected by the contextual bandit.
    - update_rate_hz ∈ {0.5, 1.0, 2.0}
    - offload=True → use edge-provided semantics; False → use local fallback
    - saccade_gating=True → enable gating; False → disable
    """
    update_rate_hz: Literal[0.5, 1.0, 2.0]
    offload: bool
    saccade_gating: bool


@dataclass(frozen=True)
class ABRDecision:
    """
    Tile->bitrate assignment for a stage (prefetch/enhance) under a budget.
    Contract:
    - selected maps TileId to a valid BitrateLevel index.
    - budget_kbps is the budget used for this decision.
    """
    stage: Literal["prefetch", "enhance"]
    selected: Mapping[TileId, BitrateLevel]
    budget_kbps: Kbps


# ---------- Context passed into the bandit ----------

@dataclass(frozen=True)
class HeadVelocityStats:
    mean_rad_s: float
    std_rad_s: float
    p75_rad_s: float


@dataclass(frozen=True)
class BanditContext:
    """
    Minimal sufficient statistics for the bandit decision.
    """
    t_ms: Ms
    bw_ewma_kbps: float       # smoothed bandwidth estimate
    bw_var_kbps2: float       # recent bandwidth variance
    rtt_ms: float
    buffer_s: float
    head_vel: HeadVelocityStats
    pred_entropy: float       # from last predictor call
    last_action: Optional[BanditAction]


# ---------- Logging schema (for JSONL) ----------

class TickLog(TypedDict, total=False):
    """
    One row in the time-stepped log emitted by the player.
    Required fields are kept minimal; modules can append extra keys.
    """
    t_ms: int
    clip: str
    user: str
    yaw_rad: float
    pitch_rad: float
    decision_latency_ms: float
    bandwidth_kbps: int
    rtt_ms: float
    buffer_s: float
    bandit: Dict[str, object]       # e.g., {"update_rate_hz": 1.0, "offload": true, ...}
    abr: Dict[str, object]          # e.g., {"selected": {"34": 2}, "budget_kbps": 2000}
    net_profile: str
    event: str


# ---------- Module interfaces (Protocols) ----------
# These define the functions each component MUST provide.
# Concrete implementations live in their own packages.

@runtime_checkable
class FoVPredictor(Protocol):
    """
    Tiny cross-modal predictor (1–3M params).
    Must be fast enough that predict(...) can be used twice per chunk
    (prefetch and near-deadline horizons) while staying ≤ 20 ms total control-path latency.
    """
    def predict(
        self,
        head: HeadMotionHistory,
        fused_semantics: FusedSemanticMap,
        horizon_ms: Ms,
    ) -> PredictedFoV:
        ...


@runtime_checkable
class PersonalizationModel(Protocol):
    """
    Learns/updates per-user class weights w from short history (≥ 2 min).
    Keeps all personal data on-device.
    """
    def fit(
        self,
        classes: Sequence[str],
        viewing_events: Iterable[Tuple[Ms, Sequence[TileId]]],
    ) -> PersonalizationWeights:
        """
        viewing_events: iterable of (t_ms, tiles_in_viewport) pairs.
        """
        ...

    def fuse(
        self,
        sem_map: SemanticMap,
        weights: PersonalizationWeights,
    ) -> FusedSemanticMap:
        """
        Apply user weights to the semantic map to produce a personalized map.
        Typically softmax(w ⊙ P_sem) or equivalent normalization.
        """
        ...


@runtime_checkable
class BanditController(Protocol):
    """
    Contextual bandit selecting (update_rate, offload/local, gating on/off).
    """
    def choose(self, ctx: BanditContext) -> BanditAction:
        ...

    def observe(self, ctx: BanditContext, action: BanditAction, reward: float) -> None:
        """
        reward = α * VWS-PSNR  - β * rebuffer  - γ * E_perception/pred
        NOTE: scaling/normalization handled by caller; reward is higher-is-better.
        """
        ...


@runtime_checkable
class ABRAlgorithm(Protocol):
    """
    Two-stage tiling/bitrate selector (MCKP-like).
    """
    def select_tiles(
        self,
        stage: Literal["prefetch", "enhance"],
        budget_kbps: Kbps,
        pred_fov: PredictedFoV,
        fused_semantics: FusedSemanticMap,
    ) -> ABRDecision:
        ...


@runtime_checkable
class SemanticsProvider(Protocol):
    """
    Produces low-frequency semantic priors (edge-offloaded in our prototype via precompute).
    """
    def get_map(self, t_ms: Ms) -> SemanticMap:
        ...


@runtime_checkable
class DatasetLoader(Protocol):
    """
    Standardizes access to head traces and (optionally) gaze/video metadata per clip/user.
    """
    def list_users(self, clip: str) -> Sequence[str]:
        ...

    def head_history(
        self, clip: str, user: str, t_start: Ms, t_end: Ms
    ) -> HeadMotionHistory:
        ...

# ---------- Lightweight validators (optional runtime checks) ----------

def assert_tile_id(tile: TileId) -> None:
    if not (0 <= int(tile) < NUM_TILES):
        raise ValueError(f"TileId out of range: {tile}")

def assert_semantic_map_shape(m: SemanticMap | FusedSemanticMap) -> None:
    K = len(m.classes)
    if len(m.weights) != K:
        raise ValueError("SemanticMap: K dimension mismatch (classes vs weights)")
    if any(len(row) != TILE_COLS for k in m.weights for row in k):
        raise ValueError("SemanticMap: expected TILE_COLS per row")
    if any(len(k) != TILE_ROWS for k in m.weights):
        raise ValueError("SemanticMap: expected TILE_ROWS per class")

def assert_head_history(h: HeadMotionHistory) -> None:
    if len(h.samples) < 2:
        raise ValueError("HeadMotionHistory: need at least 2 samples")
    last_t = -10**9
    for s in h.samples:
        if int(s.t_ms) <= last_t:
            raise ValueError("HeadMotionHistory: timestamps must be strictly increasing")
        last_t = int(s.t_ms)

# ---------- Example factory helpers (for tests / stubs) ----------

def empty_fused_map(t_ms: Ms, classes: Sequence[str]) -> FusedSemanticMap:
    """Uniform weights across tiles/classes; handy for initial stubs."""
    K = len(classes)
    w = [[[1.0 for _ in range(TILE_COLS)] for _ in range(TILE_ROWS)] for _ in range(K)]
    return FusedSemanticMap(t_ms=t_ms, classes=tuple(classes), weights=w)

def noop_bandit_action() -> BanditAction:
    return BanditAction(update_rate_hz=1.0, offload=True, saccade_gating=False)
