"""
Social Issue Taxonomy for MetaUrban Social Reward Learning

Defines the canonical set of social issue labels, severity levels, confidence
standards, and per-label penalty weights used throughout the offline annotation
pipeline and reward computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Severity levels
# ---------------------------------------------------------------------------

class SeverityLevel(IntEnum):
    """
    Unified severity scale used by every social issue label.

    0 = None      -- no issue detected
    1 = Mild      -- minor deviation, unlikely to cause incident
    2 = Moderate  -- clear violation, may affect others noticeably
    3 = Severe    -- dangerous, likely causes incident or strong discomfort
    """
    NONE     = 0
    MILD     = 1
    MODERATE = 2
    SEVERE   = 3


SEVERITY_DESCRIPTIONS: Dict[SeverityLevel, str] = {
    SeverityLevel.NONE:     "No issue",
    SeverityLevel.MILD:     "Minor deviation, no direct threat",
    SeverityLevel.MODERATE: "Clear violation, affects others",
    SeverityLevel.SEVERE:   "Dangerous, causes incident or strong discomfort",
}


# ---------------------------------------------------------------------------
# Confidence levels
# ---------------------------------------------------------------------------

class ConfidenceLevel(IntEnum):
    """
    Annotation confidence level reported by the LLM.

    1 = Low    -- marginal evidence, should be human-reviewed
    2 = Medium -- reasonable evidence but some ambiguity
    3 = High   -- clear, unambiguous evidence in the clip
    """
    LOW    = 1
    MEDIUM = 2
    HIGH   = 3


# Samples with confidence below this threshold go to the human review queue.
CONFIDENCE_REVIEW_THRESHOLD: int = ConfidenceLevel.MEDIUM


# ---------------------------------------------------------------------------
# Social issue label definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SocialIssueLabel:
    """
    Descriptor for one social issue label.

    Attributes
    ----------
    name_en  : English short identifier used in code.
    name_zh  : Chinese display name (data field).
    desc_en  : Full English description.
    desc_zh  : Full Chinese description (data field).
    penalty_weight        : Default penalty weight in the linear reward formula (lambda * w_k).
    requires_pedestrian   : If True, the issue is invalid when no pedestrian is present.
    requires_moving_agent : If True, the issue is invalid when ego is stationary.
    severity_thresholds   : (mild_lower, moderate_lower, severe_lower) -- lower bounds
                            of the observable metric defining each severity tier.
                            Units are label-specific; see desc_en.
    """
    name_en: str
    name_zh: str
    desc_en: str
    desc_zh: str
    penalty_weight: float
    requires_pedestrian: bool = False
    requires_moving_agent: bool = False
    severity_thresholds: Tuple[float, float, float] = (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Canonical label set (8 labels)
# ---------------------------------------------------------------------------

SOCIAL_ISSUE_DEFINITIONS: Dict[str, SocialIssueLabel] = {

    # L1
    "failure_to_yield": SocialIssueLabel(
        name_en="failure_to_yield",
        name_zh="未礼让行人",
        desc_en=(
            "Ego agent does not yield to a pedestrian who has legal right-of-way "
            "(crosswalk or conflict zone). Measured by time-to-collision (TTC) with "
            "the nearest pedestrian at a conflict point."
        ),
        desc_zh=(
            "自车未向具有法定优先通行权的行人（人行横道或冲突区域）礼让。"
            "以冲突点处最近行人的碰撞时间（TTC）为量化指标。"
        ),
        penalty_weight=2.0,
        requires_pedestrian=True,
        requires_moving_agent=True,
        # TTC thresholds in seconds: mild < 4s, moderate < 2s, severe < 1s
        severity_thresholds=(4.0, 2.0, 1.0),
    ),

    # L2
    "unsafe_proximity": SocialIssueLabel(
        name_en="unsafe_proximity",
        name_zh="近距压迫",
        desc_en=(
            "Ego agent maintains an uncomfortably small lateral or longitudinal gap "
            "to another agent (vehicle, pedestrian, or cyclist), creating intimidation "
            "or a collision hazard. Measured by minimum distance to nearest agent."
        ),
        desc_zh=(
            "自车与其他参与者（车辆、行人或骑手）保持过小的横向或纵向间距，"
            "产生压迫感或碰撞风险。以最近参与者的最小距离为量化指标。"
        ),
        penalty_weight=1.5,
        requires_pedestrian=False,
        requires_moving_agent=False,
        # Distance thresholds in meters: mild < 2m, moderate < 1m, severe < 0.5m
        severity_thresholds=(2.0, 1.0, 0.5),
    ),

    # L3
    "blocking_passage": SocialIssueLabel(
        name_en="blocking_passage",
        name_zh="阻塞通行",
        desc_en=(
            "Ego agent unnecessarily occupies a key passage (sidewalk, crosswalk, "
            "bike lane, or doorway), preventing or significantly delaying others. "
            "Measured by dwell time in a restricted zone without making progress."
        ),
        desc_zh=(
            "自车不必要地占用关键通道（人行道、人行横道、自行车道或门口），"
            "阻止或显著延误他人通行。以在受限区域内停留时间为量化指标。"
        ),
        penalty_weight=1.2,
        requires_pedestrian=False,
        requires_moving_agent=False,
        # Dwell-time thresholds in seconds: mild > 2s, moderate > 5s, severe > 10s
        severity_thresholds=(2.0, 5.0, 10.0),
    ),

    # L4
    "sudden_aggression": SocialIssueLabel(
        name_en="sudden_aggression",
        name_zh="突然切入/急加速",
        desc_en=(
            "Ego agent performs an abrupt lane change, sudden acceleration, or "
            "unexpected cut-in that forces another agent to brake or swerve. "
            "Measured by induced deceleration of the affected agent (m/s^2)."
        ),
        desc_zh=(
            "自车进行突然变道、急加速或意外切入，迫使其他参与者急刹或躲避。"
            "以被影响参与者的诱发减速度（m/s²）为量化指标。"
        ),
        penalty_weight=1.8,
        requires_pedestrian=False,
        requires_moving_agent=True,
        # Induced deceleration thresholds in m/s^2: mild > 1, moderate > 3, severe > 5
        severity_thresholds=(1.0, 3.0, 5.0),
    ),

    # L5
    "inappropriate_speed_near_crowd": SocialIssueLabel(
        name_en="inappropriate_speed_near_crowd",
        name_zh="人群附近速度不当",
        desc_en=(
            "Ego agent travels at an unsafe speed in areas with high pedestrian "
            "density or near vulnerable road users (children, wheelchair users, "
            "elderly). Measured by speed x crowd_density index."
        ),
        desc_zh=(
            "自车在行人密集区域或弱势参与者（儿童、轮椅使用者、老年人）附近"
            "以不安全速度行驶。以速度×人流密度指数为量化指标。"
        ),
        penalty_weight=1.5,
        requires_pedestrian=True,
        requires_moving_agent=True,
        # Speed-density index thresholds: mild > 1.0, moderate > 2.0, severe > 4.0
        severity_thresholds=(1.0, 2.0, 4.0),
    ),

    # L6
    "vulnerable_user_close_pass": SocialIssueLabel(
        name_en="vulnerable_user_close_pass",
        name_zh="近距驶过弱势参与者",
        desc_en=(
            "Ego agent passes dangerously close to a vulnerable road user "
            "(wheelchair, child, elderly, visually impaired) without reducing speed. "
            "Measured by min lateral gap at pass moment."
        ),
        desc_zh=(
            "自车在未减速的情况下极近地驶过弱势参与者（轮椅、儿童、老年人、"
            "视障人士）。以超车时刻的最小横向间距为量化指标。"
        ),
        penalty_weight=2.5,
        requires_pedestrian=True,
        requires_moving_agent=True,
        # Lateral gap thresholds in meters: mild < 1.5m, moderate < 0.8m, severe < 0.4m
        severity_thresholds=(1.5, 0.8, 0.4),
    ),

    # L7
    "ignoring_social_signal": SocialIssueLabel(
        name_en="ignoring_social_signal",
        name_zh="忽视社交信号",
        desc_en=(
            "Ego agent ignores an explicit social cue from another agent: a pedestrian "
            "waving to cross, a cyclist signaling a turn, or a disability-indicating "
            "marking. Measured by IoU-overlap of agent gaze/gesture bounding box "
            "with ego's predicted attention region."
        ),
        desc_zh=(
            "自车忽视其他参与者的明确社交信号：行人挥手示意过马路、骑手打手势"
            "转弯，或障碍标识。以参与者手势/视线边界框与自车预测注意区域的IoU重叠度"
            "为量化指标。"
        ),
        penalty_weight=1.3,
        requires_pedestrian=True,
        requires_moving_agent=False,
        # IoU-overlap thresholds: mild > 0.3, moderate > 0.5, severe > 0.7
        severity_thresholds=(0.3, 0.5, 0.7),
    ),

    # L8
    "lane_encroachment": SocialIssueLabel(
        name_en="lane_encroachment",
        name_zh="侵占他人车道/通道",
        desc_en=(
            "Ego agent encroaches on a lane or zone designated for other users "
            "(bike lane, bus stop, pedestrian-only zone), disrupting their flow. "
            "Measured by overlap proportion with the forbidden zone and dwell time."
        ),
        desc_zh=(
            "自车侵占专用于其他参与者的车道或区域（自行车道、公交停靠站、"
            "步行专用区），破坏其正常通行流。以与禁区的重叠比例及停留时间为量化指标。"
        ),
        penalty_weight=1.0,
        requires_pedestrian=False,
        requires_moving_agent=False,
        # Overlap-duration product thresholds in seconds: mild > 1s, moderate > 3s, severe > 6s
        severity_thresholds=(1.0, 3.0, 6.0),
    ),
}


# ---------------------------------------------------------------------------
# Per-label penalty weights convenience dict
# ---------------------------------------------------------------------------

LABEL_WEIGHTS: Dict[str, float] = {
    name: lbl.penalty_weight
    for name, lbl in SOCIAL_ISSUE_DEFINITIONS.items()
}


# ---------------------------------------------------------------------------
# Annotation JSON schema
# ---------------------------------------------------------------------------

#: Canonical JSON schema that every LLM annotation must conform to.
ANNOTATION_JSON_SCHEMA: Dict = {
    "type": "object",
    "required": [
        "clip_id", "label", "present", "severity", "confidence",
        "start_frame", "end_frame", "evidence",
    ],
    "properties": {
        "clip_id":     {"type": "string",  "description": "Unique clip identifier"},
        "label":       {"type": "string",  "enum": list(SOCIAL_ISSUE_DEFINITIONS.keys())},
        "present":     {"type": "boolean", "description": "Whether the issue is observed"},
        "severity":    {"type": "integer", "minimum": 0, "maximum": 3,
                        "description": "SeverityLevel: 0=None, 1=Mild, 2=Moderate, 3=Severe"},
        "confidence":  {"type": "integer", "minimum": 1, "maximum": 3,
                        "description": "ConfidenceLevel: 1=Low, 2=Medium, 3=High"},
        "start_frame": {"type": "integer", "minimum": 0,
                        "description": "First frame where the issue begins (0-indexed)"},
        "end_frame":   {"type": "integer", "minimum": 0,
                        "description": "Last frame where the issue is active (0-indexed)"},
        "evidence":    {"type": "string",
                        "description": "Short textual evidence description (<=120 chars)"},
    },
}


# ---------------------------------------------------------------------------
# LLM system prompt template
# ---------------------------------------------------------------------------

LLM_SYSTEM_PROMPT: str = """
You are a social behavior safety analyst for autonomous robots navigating urban pedestrian environments.

Your task is to watch a short video clip and evaluate it for the following social issue labels:

{label_block}

For EACH label, output a JSON object matching this schema exactly:
  clip_id        : string  -- provided in the user message
  label          : string  -- one of the label names listed above
  present        : boolean -- true if the issue occurs at any point in the clip
  severity       : integer -- 0 (None) / 1 (Mild) / 2 (Moderate) / 3 (Severe)
  confidence     : integer -- 1 (Low) / 2 (Medium) / 3 (High)
  start_frame    : integer -- first frame where the issue begins (0-indexed)
  end_frame      : integer -- last frame where the issue is active (0-indexed)
  evidence       : string  -- <=120 chars describing the observable evidence

Rules:
1. If `present` is false, set severity=0, confidence=3, start_frame=0, end_frame=0, evidence="N/A".
2. Do NOT mark failure_to_yield or inappropriate_speed_near_crowd if NO pedestrian is visible.
3. Set confidence=1 whenever the evidence is ambiguous or partially occluded -- these will be human-reviewed.
4. Output a JSON array containing one object per label. No extra keys, no Markdown, no prose.
""".strip()


def _build_label_block() -> str:
    lines = []
    for name, lbl in SOCIAL_ISSUE_DEFINITIONS.items():
        lines.append(f"  - {name}: {lbl.desc_en}")
    return "\n".join(lines)


#: Ready-to-use system prompt with all label descriptions injected.
LLM_SYSTEM_PROMPT_FORMATTED: str = LLM_SYSTEM_PROMPT.format(
    label_block=_build_label_block()
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def get_label_by_name(name: str) -> SocialIssueLabel:
    """Return the SocialIssueLabel for *name*, or raise KeyError."""
    if name not in SOCIAL_ISSUE_DEFINITIONS:
        raise KeyError(
            f"Unknown social issue label: '{name}'. "
            f"Valid labels: {list(SOCIAL_ISSUE_DEFINITIONS.keys())}"
        )
    return SOCIAL_ISSUE_DEFINITIONS[name]


def infer_severity(label_name: str, metric_value: float) -> SeverityLevel:
    """
    Map a raw metric value to a SeverityLevel using the label's thresholds.

    For labels where a *lower* metric value is worse (TTC, distance, gap),
    thresholds are stored in decreasing order and the rule is
    ``value < threshold``.

    For labels where a *higher* metric value is worse (dwell time,
    speed-density index, encroachment duration), thresholds are stored in
    increasing order and the rule is ``value > threshold``.

    Direction is inferred automatically from whether the threshold sequence
    is non-increasing (lower-is-worse) or strictly increasing (higher-is-worse).
    """
    lbl = get_label_by_name(label_name)
    t0, t1, t2 = lbl.severity_thresholds
    if t0 >= t1 >= t2:
        # Lower metric value is worse (e.g., TTC, distance, gap)
        if metric_value < t2:
            return SeverityLevel.SEVERE
        if metric_value < t1:
            return SeverityLevel.MODERATE
        if metric_value < t0:
            return SeverityLevel.MILD
    else:
        # Higher metric value is worse (e.g., dwell time, speed-density)
        if metric_value > t2:
            return SeverityLevel.SEVERE
        if metric_value > t1:
            return SeverityLevel.MODERATE
        if metric_value > t0:
            return SeverityLevel.MILD
    return SeverityLevel.NONE


def validate_annotation(
    ann: dict,
    has_pedestrian: bool = True,
    ego_is_moving: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Validate a single annotation dict against field and consistency rules.

    Returns
    -------
    (is_valid, errors)
        is_valid : True when no errors are found.
        errors   : List of human-readable error messages.
    """
    errors: List[str] = []
    label_name = ann.get("label", "")

    # Required field presence
    required = {"clip_id", "label", "present", "severity",
                 "confidence", "start_frame", "end_frame", "evidence"}
    missing = required - ann.keys()
    if missing:
        errors.append(f"Missing required fields: {missing}")
        return False, errors

    # Label validity
    if label_name not in SOCIAL_ISSUE_DEFINITIONS:
        errors.append(f"Unknown label '{label_name}'")
        return False, errors

    lbl = SOCIAL_ISSUE_DEFINITIONS[label_name]
    present  = ann["present"]
    severity = ann["severity"]
    conf     = ann["confidence"]

    if severity not in range(4):
        errors.append(f"severity={severity} out of [0, 3]")

    if conf not in range(1, 4):
        errors.append(f"confidence={conf} out of [1, 3]")

    if not present and severity != 0:
        errors.append("present=False but severity != 0")

    if present and lbl.requires_pedestrian and not has_pedestrian:
        errors.append(
            f"Label '{label_name}' requires a pedestrian in the scene "
            "but has_pedestrian=False"
        )

    if present and lbl.requires_moving_agent and not ego_is_moving:
        errors.append(
            f"Label '{label_name}' requires ego to be moving "
            "but ego_is_moving=False"
        )

    if ann["start_frame"] > ann["end_frame"]:
        errors.append(
            f"start_frame={ann['start_frame']} > end_frame={ann['end_frame']}"
        )

    if len(ann.get("evidence", "")) > 120:
        errors.append("evidence exceeds 120 characters")

    return len(errors) == 0, errors


def compute_social_penalty(
    annotations: List[dict],
    global_lambda: float = 0.5,
    custom_weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute the total social penalty for a clip from its annotation list.

    Formula::

        penalty = lambda * sum_k ( w_k * severity_k / 3 )

    Parameters
    ----------
    annotations   : List of validated annotation dicts.
    global_lambda : Global scaling factor (default 0.5).
    custom_weights: Per-label weight overrides; falls back to LABEL_WEIGHTS.

    Returns
    -------
    (total_penalty, breakdown)
        total_penalty : Scalar float.
        breakdown     : Dict mapping label name to its weighted contribution.
    """
    weights = custom_weights or LABEL_WEIGHTS
    breakdown: Dict[str, float] = {}
    total = 0.0
    for ann in annotations:
        if not ann.get("present", False):
            continue
        name    = ann["label"]
        sev     = ann.get("severity", 0)
        w       = weights.get(name, 1.0)
        contrib = global_lambda * w * (sev / 3.0)
        breakdown[name] = contrib
        total += contrib
    return total, breakdown


# ---------------------------------------------------------------------------
# Quick self-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Social Issue Taxonomy ===\n")
    for name, lbl in SOCIAL_ISSUE_DEFINITIONS.items():
        print(f"[{name}]")
        print(f"  weight={lbl.penalty_weight}  thresholds={lbl.severity_thresholds}")
        print(f"  {lbl.desc_en[:90]}...")
        print()

    print("=== Severity inference ===")
    print("failure_to_yield, TTC=0.8s  ->", infer_severity("failure_to_yield", 0.8))
    print("unsafe_proximity, dist=1.2m ->", infer_severity("unsafe_proximity", 1.2))
    print("blocking_passage, dwell=7s  ->", infer_severity("blocking_passage", 7.0))

    print("\n=== Annotation validation ===")
    sample = {
        "clip_id": "clip_001",
        "label": "failure_to_yield",
        "present": True,
        "severity": 2,
        "confidence": 3,
        "start_frame": 10,
        "end_frame": 45,
        "evidence": "Ego passes crosswalk at 4m/s with pedestrian 0.8s ahead",
    }
    valid, errs = validate_annotation(sample, has_pedestrian=True, ego_is_moving=True)
    print("Valid (with pedestrian)?", valid, "| Errors:", errs)

    valid2, errs2 = validate_annotation(sample, has_pedestrian=False)
    print("Valid (no pedestrian)?  ", valid2, "| Errors:", errs2)

    print("\n=== Penalty computation ===")
    anns = [
        sample,
        {
            "clip_id": "clip_001", "label": "unsafe_proximity", "present": True,
            "severity": 1, "confidence": 2, "start_frame": 20, "end_frame": 30,
            "evidence": "1.4m lateral gap when passing cyclist",
        },
    ]
    total, bd = compute_social_penalty(anns, global_lambda=0.5)
    print(f"Total penalty: {total:.4f}")
    for k, v in bd.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== LLM System Prompt (first 400 chars) ===")
    print(LLM_SYSTEM_PROMPT_FORMATTED[:400], "...")
