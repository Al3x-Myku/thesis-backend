"""Evaluation / human-validation router.

Two access tiers:
  • Researcher  (JWT)  — create study sessions, list results, export CSV.
  • Participant (token) — no account; access via a UUID token the researcher shares.

Participant endpoints are deliberately open (no JWT) — the token IS the credential.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import PlainTextResponse, HTMLResponse
from pydantic import BaseModel, Field, field_validator
from sqlmodel import Session

from app.core.security import get_current_user
from app.database import get_db
from app.models.user import User
from app.services import evaluation_service

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PYDANTIC SCHEMAS  (the actual form definitions)                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Researcher: create a study session ───────────────────────────────────────

class CreateStudySessionRequest(BaseModel):
    scene_id: int
    design_session_id: Optional[int] = None
    generation_engine_disclosed: bool = False
    notes: Optional[str] = None
    expires_at: Optional[datetime] = None


class StudySessionResponse(BaseModel):
    id: int
    token: str
    scene_id: int
    design_session_id: Optional[int]
    generation_engine_disclosed: bool
    notes: Optional[str]
    created_at: datetime
    expires_at: Optional[datetime]
    participant_url_hint: str

    model_config = {"from_attributes": True}


# ── Form 1: Evaluator profile (demographics) ─────────────────────────────────

class EvaluatorProfileForm(BaseModel):
    """
    Background questionnaire — filled once per participant before seeing any variants.
    All fields are required unless marked Optional.
    """
    participant_code: str = Field(
        ...,
        description=(
            "Anonymous identifier assigned by the researcher (e.g. 'P001'). "
            "You will need this code to submit all subsequent forms."
        ),
    )
    age_range: Literal["<25", "25-34", "35-44", "45-54", "55+"] = Field(
        ..., description="Your age bracket."
    )
    gender: Optional[str] = Field(
        default=None,
        description="Optional — self-described or left blank.",
        max_length=80,
    )
    professional_role: Literal[
        "student",
        "practicing_architect",
        "interior_designer",
        "researcher",
        "academic",
        "other",
    ] = Field(..., description="Your primary professional role.")
    professional_role_other: Optional[str] = Field(
        default=None,
        description="If 'other', please specify.",
        max_length=200,
    )
    years_experience: Literal["0-2", "3-5", "6-10", "11-20", ">20"] = Field(
        ..., description="Years of professional experience in architecture or design."
    )
    education_level: Literal[
        "bachelors", "masters", "phd", "professional_license", "other"
    ] = Field(..., description="Highest completed level of education.")
    education_level_other: Optional[str] = Field(
        default=None, description="If 'other', specify.", max_length=200
    )
    specialization: Literal[
        "residential",
        "commercial",
        "hospitality",
        "urban_planning",
        "landscape",
        "heritage_conservation",
        "multiple",
        "other",
    ] = Field(..., description="Primary design specialization.")
    specialization_other: Optional[str] = Field(
        default=None, description="If 'other', specify.", max_length=200
    )
    country_of_education: str = Field(
        ..., description="Country where you completed your primary design education.", max_length=100
    )
    software_used: List[str] = Field(
        default_factory=list,
        description=(
            "Design/visualization software you use regularly. "
            "Examples: AutoCAD, Revit, ArchiCAD, SketchUp, Rhino, Blender, Lumion, "
            "Enscape, V-Ray, 3ds Max, Vectorworks, Adobe CC, Midjourney, DALL-E."
        ),
    )
    familiarity_ai_tools: int = Field(
        ...,
        ge=1,
        le=5,
        description=(
            "How familiar are you with AI-powered design tools? "
            "1 = Never heard of them  "
            "2 = Heard of them but never used  "
            "3 = Tried them occasionally  "
            "4 = Use them regularly  "
            "5 = Use them daily / expert user"
        ),
    )
    familiarity_3d_modeling: int = Field(
        ...,
        ge=1,
        le=5,
        description=(
            "How proficient are you with 3D modeling software? "
            "1 = No experience  2 = Beginner  3 = Intermediate  "
            "4 = Advanced  5 = Expert"
        ),
    )
    familiarity_interior_design_software: int = Field(
        ...,
        ge=1,
        le=5,
        description=(
            "How proficient are you with interior-design-specific software "
            "(e.g. Roomstyler, Planner5D, DesignApp)? "
            "1 = No experience  …  5 = Expert"
        ),
    )
    has_used_ai_design_tools_before: bool = Field(
        ...,
        description="Have you used any AI-assisted design or rendering tool before this study?",
    )
    ai_tools_used_names: Optional[str] = Field(
        default=None,
        description="If yes above: which tools have you used? (free text)",
        max_length=500,
    )

    @field_validator("software_used", mode="before")
    @classmethod
    def _coerce_software(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v


class EvaluatorProfileResponse(BaseModel):
    id: int
    study_session_id: int
    participant_code: str
    submitted_at: datetime

    model_config = {"from_attributes": True}


# ── Form 2: Per-variant 2D evaluation ────────────────────────────────────────

_LIKERT_7 = Field(..., ge=1, le=7)

class Variant2DEvaluationForm(BaseModel):
    """
    Rate a single 2D generated room variant.
    All Likert scales are 1–7 unless otherwise noted.
    Fill this form once for EACH variant you are shown.
    """
    participant_code: str = Field(..., description="Your anonymous participant code (from the demographics form).")
    engine_type: Literal["diffusion", "baseline"] = Field(
        default="diffusion",
        description="Which generation method produced this variant (filled by system if not known).",
    )

    # ── Appearance / palette channel ─────────────────────────────────────────
    aesthetic_quality: int = Field(
        **_LIKERT_7.field_info.__dict__,
        ge=1, le=7,
        description=(
            "Overall aesthetic quality of the room design. "
            "1 = Very poor  4 = Acceptable  7 = Excellent"
        ),
    )
    color_harmony: int = Field(
        ge=1, le=7,
        description=(
            "How harmoniously do the colors in this room work together? "
            "1 = Clashing / discordant  7 = Perfectly harmonious"
        ),
    )
    palette_fidelity_to_moodboard: int = Field(
        ge=1, le=7,
        description=(
            "How well does the room's color palette reflect the mood board you were shown? "
            "1 = No resemblance  7 = Perfectly faithful to the mood board"
        ),
    )
    atmosphere_mood: int = Field(
        ge=1, le=7,
        description=(
            "How successfully does this design evoke the atmosphere/mood intended by the mood board? "
            "1 = Not at all  7 = Exactly as intended"
        ),
    )
    color_temperature_appropriateness: int = Field(
        ge=1, le=7,
        description=(
            "Is the warm/cool balance of the color temperature appropriate for this type of space? "
            "1 = Very inappropriate  7 = Perfectly appropriate"
        ),
    )

    # ── Structure / furniture channel ────────────────────────────────────────
    spatial_layout_preservation: int = Field(
        ge=1, le=7,
        description=(
            "How well is the original room's spatial layout (walls, windows, doors, openings) preserved? "
            "1 = Completely changed  7 = Identical layout"
        ),
    )
    furniture_style_consistency: int = Field(
        ge=1, le=7,
        description=(
            "Do the furniture pieces feel stylistically coherent with each other and the room? "
            "1 = Completely inconsistent  7 = Perfectly coherent"
        ),
    )
    furniture_placement_plausibility: int = Field(
        ge=1, le=7,
        description=(
            "Are the furniture pieces placed in physically and functionally plausible positions? "
            "1 = Impossible / bizarre placements  7 = Natural, functional placement"
        ),
    )
    scale_and_proportion_accuracy: int = Field(
        ge=1, le=7,
        description=(
            "Do the furniture pieces appear correctly scaled relative to the room and each other? "
            "1 = Severely disproportionate  7 = Accurately proportioned"
        ),
    )

    # ── Realism / quality ────────────────────────────────────────────────────
    photorealism: int = Field(
        ge=1, le=7,
        description=(
            "How photorealistic does the image look overall? "
            "1 = Clearly computer-generated / cartoonish  7 = Indistinguishable from a real photograph"
        ),
    )
    lighting_plausibility: int = Field(
        ge=1, le=7,
        description=(
            "How physically plausible is the lighting in the scene? "
            "1 = Completely unrealistic  7 = Fully convincing lighting"
        ),
    )
    material_surface_quality: int = Field(
        ge=1, le=7,
        description=(
            "How convincing are the material textures and surface finishes (wood grain, fabric, tile, etc.)? "
            "1 = Fake / blurry textures  7 = Convincing, high-quality surfaces"
        ),
    )
    shadow_and_reflection_quality: int = Field(
        ge=1, le=7,
        description=(
            "How realistic are the shadows, reflections, and ambient occlusion? "
            "1 = Absent or wrong  7 = Physically accurate"
        ),
    )

    # ── Professional applicability ───────────────────────────────────────────
    professional_suitability: int = Field(
        ge=1, le=7,
        description=(
            "How suitable is this output for professional use in an architecture/design practice? "
            "1 = Unusable professionally  7 = Ready for direct professional use"
        ),
    )
    client_presentability: int = Field(
        ge=1, le=7,
        description=(
            "Would you show this image to a client as a design proposal? "
            "1 = Definitely not  7 = Absolutely, without modification"
        ),
    )
    innovation_and_creativity: int = Field(
        ge=1, le=7,
        description=(
            "How novel or creatively interesting is the design solution? "
            "1 = Generic / uninspired  7 = Highly innovative"
        ),
    )
    design_coherence: int = Field(
        ge=1, le=7,
        description=(
            "Does the overall design feel like a unified, intentional concept? "
            "1 = Random / incoherent  7 = Fully unified design concept"
        ),
    )

    # ── Categorical ──────────────────────────────────────────────────────────
    would_present_to_client: Literal["yes", "no", "maybe"] = Field(
        ..., description="Would you present this variant to a real client as a design option?"
    )
    manual_redesign_time_estimate: Literal["<1h", "1-4h", "4-8h", "1-3d", ">3d"] = Field(
        ...,
        description=(
            "Approximately how long would it take you to produce an equivalent result manually? "
            "'<1h' = under 1 hour, '1-4h', '4-8h', '1-3d' = 1 to 3 days, '>3d' = more than 3 days."
        ),
    )

    # ── Open text ────────────────────────────────────────────────────────────
    most_appealing_aspect: Optional[str] = Field(
        default=None,
        description=(
            "What is the single most appealing aspect of this design? "
            "Be specific (e.g. 'the warm terracotta tones contrast beautifully with the light oak flooring')."
        ),
        max_length=1000,
    )
    most_problematic_aspect: Optional[str] = Field(
        default=None,
        description=(
            "What is the most significant problem or weakness in this variant? "
            "Be as specific as possible."
        ),
        max_length=1000,
    )
    design_suggestions: Optional[str] = Field(
        default=None,
        description=(
            "As a design professional, what specific changes would you recommend to improve this variant? "
            "Consider: color, materials, furniture selection, scale, layout, lighting, style coherence."
        ),
        max_length=2000,
    )
    elements_that_look_ai_generated: Optional[str] = Field(
        default=None,
        description=(
            "Which specific elements, if any, make it apparent this image was AI-generated "
            "(e.g. 'the upholstery texture looks unrealistically uniform', 'window reflections are wrong')?"
        ),
        max_length=1000,
    )


class Variant2DEvalResponse(BaseModel):
    id: int
    variant_id: int
    evaluator_profile_id: int
    submitted_at: datetime

    model_config = {"from_attributes": True}


# ── Form 3: Variant set comparison (cross-variant) ───────────────────────────

class VariantSetEvaluationForm(BaseModel):
    """
    Comparative evaluation — complete this AFTER rating all individual variants.
    This form probes the core research question: can palette and furniture be edited independently?
    """
    participant_code: str = Field(..., description="Your anonymous participant code.")

    # ── Ranking ──────────────────────────────────────────────────────────────
    variant_ranking: List[int] = Field(
        ...,
        description=(
            "Rank the variants from BEST to WORST by entering their IDs in order. "
            "Example: [3, 1, 4, 2] means variant 3 is best, variant 2 is worst."
        ),
    )
    preferred_variant_id: Optional[int] = Field(
        default=None,
        description="ID of the variant you would most want to develop further (your top choice).",
    )
    preferred_variant_reason: Optional[str] = Field(
        default=None,
        description=(
            "Why is this your preferred variant? What specifically makes it stand out? "
            "Explain in terms of palette, layout, atmosphere, or professional suitability."
        ),
        max_length=1000,
    )

    # ── Disentanglement perception (core research probe) ─────────────────────
    palette_change_perceived: bool = Field(
        ...,
        description=(
            "Looking across all the variants you were shown: "
            "did you perceive the COLOR PALETTE changing between variants?"
        ),
    )
    furniture_change_perceived: bool = Field(
        ...,
        description=(
            "Did you perceive the FURNITURE LAYOUT or FURNITURE STYLE changing between variants?"
        ),
    )
    perceived_what_changed: Dict[str, bool] = Field(
        default_factory=dict,
        description=(
            "For each category, indicate whether you noticed it changing across variants. "
            "Expected keys: 'color_palette', 'furniture_identity', 'furniture_placement', "
            "'lighting', 'materials', 'spatial_layout', 'overall_atmosphere'."
        ),
    )
    disentanglement_clarity: int = Field(
        ge=1,
        le=7,
        description=(
            "How clearly were the color palette and furniture kept as SEPARATE, independently varied axes? "
            "1 = They seemed to change together / inseparable  "
            "4 = Somewhat separable  "
            "7 = Clearly and independently controllable"
        ),
    )
    palette_axis_control_confidence: int = Field(
        ge=1,
        le=7,
        description=(
            "How confident are you that color palette changes were deliberate and controlled "
            "(not random or accidental)? "
            "1 = Looks random  7 = Clearly intentional and consistent"
        ),
    )
    furniture_axis_control_confidence: int = Field(
        ge=1,
        le=7,
        description=(
            "How confident are you that furniture/layout changes were deliberate and controlled? "
            "1 = Looks random  7 = Clearly intentional and consistent"
        ),
    )
    cross_channel_leakage_observed: bool = Field(
        ...,
        description=(
            "Did you observe 'leakage' — where a change intended for one axis "
            "(e.g. color) also unexpectedly altered the other axis (e.g. furniture shape)?"
        ),
    )
    leakage_description: Optional[str] = Field(
        default=None,
        description=(
            "If you observed leakage, describe it specifically: "
            "what changed that shouldn't have, and in which variants?"
        ),
        max_length=1000,
    )

    # ── Comparative quality ───────────────────────────────────────────────────
    consistency_across_variants: int = Field(
        ge=1,
        le=7,
        description=(
            "Do all variants feel like they belong to the same underlying room? "
            "1 = Completely different rooms  7 = Clearly the same room with variations"
        ),
    )
    variation_diversity: int = Field(
        ge=1,
        le=7,
        description=(
            "Are the variants sufficiently different from each other to be useful as design alternatives? "
            "1 = All look identical  7 = Meaningfully distinct alternatives"
        ),
    )
    best_palette_match_variant_id: Optional[int] = Field(
        default=None,
        description="Which variant best matched the mood board palette? (variant ID)",
    )
    best_layout_match_variant_id: Optional[int] = Field(
        default=None,
        description="Which variant best preserved the original room's spatial layout? (variant ID)",
    )

    # ── AI tool assessment ────────────────────────────────────────────────────
    overall_ai_tool_utility: int = Field(
        ge=1,
        le=7,
        description=(
            "How useful would this AI design tool be in your professional design workflow? "
            "1 = Not useful at all  7 = Extremely useful"
        ),
    )
    overall_ai_tool_trustworthiness: int = Field(
        ge=1,
        le=7,
        description=(
            "How much would you trust AI-generated outputs like these in a real project? "
            "1 = Not at all — would never rely on it  7 = Fully trust — would rely on it directly"
        ),
    )
    workflow_integration_ease: int = Field(
        ge=1,
        le=7,
        description=(
            "How easy would it be to integrate this kind of tool into your existing design workflow? "
            "1 = Would require complete workflow redesign  7 = Fits seamlessly into existing practice"
        ),
    )
    time_saved_vs_manual: Literal["none", "<30min", "1-2h", "3-4h", ">4h"] = Field(
        ...,
        description=(
            "Compared to producing equivalent results manually, how much time would this tool save per project? "
            "Answer per design iteration cycle (not per final deliverable)."
        ),
    )
    would_use_professionally: Literal["yes", "no", "maybe", "already_do"] = Field(
        ..., description="Would you use a tool like this in your professional practice?"
    )
    tool_replaces_or_augments: Literal[
        "replaces_sketch",
        "augments_sketch",
        "replaces_rendering",
        "augments_rendering",
        "replaces_client_meeting_prep",
        "augments_client_meeting_prep",
        "other",
    ] = Field(
        ...,
        description=(
            "Which part of your workflow does this tool best fit as a replacement or augmentation? "
            "'augments' = adds to your existing process, 'replaces' = substitutes a current step."
        ),
    )
    tool_replaces_or_augments_other: Optional[str] = Field(
        default=None, description="If 'other', describe.", max_length=300
    )

    # ── Open text ─────────────────────────────────────────────────────────────
    biggest_limitation: Optional[str] = Field(
        default=None,
        description=(
            "What is the single biggest limitation of this AI design approach for real professional use? "
            "Consider output quality, control, reliability, integration, or client expectations."
        ),
        max_length=1500,
    )
    most_valuable_feature: Optional[str] = Field(
        default=None,
        description=(
            "What is the most valuable feature or capability demonstrated by this system? "
            "What would make it worth adopting despite any limitations?"
        ),
        max_length=1000,
    )
    suggested_improvements: Optional[str] = Field(
        default=None,
        description=(
            "As a design professional, what improvements would make this tool significantly more useful? "
            "Prioritize by impact: what ONE change would matter most?"
        ),
        max_length=1500,
    )
    comparison_to_existing_tools: Optional[str] = Field(
        default=None,
        description=(
            "How does this AI approach compare to visualization/rendering tools you currently use "
            "(e.g. Lumion, Enscape, V-Ray, Midjourney for moodboards)? "
            "What does it do better or worse?"
        ),
        max_length=1500,
    )
    additional_comments: Optional[str] = Field(
        default=None,
        description="Any other observations, reactions, or feedback not covered by the questions above.",
        max_length=2000,
    )

    @field_validator("variant_ranking", "perceived_what_changed", mode="before")
    @classmethod
    def _coerce_json(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v


class VariantSetEvalResponse(BaseModel):
    id: int
    study_session_id: int
    evaluator_profile_id: int
    submitted_at: datetime

    model_config = {"from_attributes": True}


# ── Form 4: 3D commit evaluation ─────────────────────────────────────────────

class Commit3DEvaluationForm(BaseModel):
    """
    Evaluation of the committed 3D GLB model.
    Complete this AFTER examining the 3D scene in the viewer.
    """
    participant_code: str = Field(..., description="Your anonymous participant code.")
    committed_variant_id: int = Field(
        ..., description="ID of the 2D variant that was committed to 3D."
    )

    # ── Geometry accuracy ────────────────────────────────────────────────────
    mesh_completeness: int = Field(
        ge=1, le=7,
        description=(
            "Are all expected room surfaces present and closed (floor, ceiling, walls, furniture)? "
            "1 = Many surfaces missing  7 = Fully complete mesh"
        ),
    )
    spatial_accuracy_dimensions: int = Field(
        ge=1, le=7,
        description=(
            "Do the room dimensions and proportions feel physically accurate? "
            "1 = Severely distorted  7 = Accurately proportioned"
        ),
    )
    furniture_geometry_accuracy: int = Field(
        ge=1, le=7,
        description=(
            "Do individual furniture objects have an accurate 3D shape? "
            "1 = Unrecognizable shapes  7 = Accurately modeled"
        ),
    )
    room_proportions_accuracy: int = Field(
        ge=1, le=7,
        description=(
            "Do the ceiling height, room width/depth, and overall spatial envelope feel correct? "
            "1 = Completely wrong  7 = Feels like a real room"
        ),
    )
    artifact_presence: int = Field(
        ge=1, le=7,
        description=(
            "Rate the ABSENCE of mesh artifacts (holes, floating polygons, intersecting geometry, z-fighting). "
            "7 = No visible artifacts  1 = Severe, distracting artifacts"
        ),
    )

    # ── Appearance / texture ─────────────────────────────────────────────────
    texture_quality: int = Field(
        ge=1, le=7,
        description=(
            "Overall quality of surface textures applied to the 3D model. "
            "1 = Blurry / missing textures  7 = Sharp, high-quality textures"
        ),
    )
    material_representation: int = Field(
        ge=1, le=7,
        description=(
            "Do material properties (roughness, reflectivity, translucency) look realistic? "
            "1 = All surfaces look the same  7 = Convincingly differentiated materials"
        ),
    )
    palette_fidelity_3d: int = Field(
        ge=1, le=7,
        description=(
            "How well does the 3D model's color palette match the 2D variant it was generated from? "
            "1 = Colors completely changed  7 = Faithful color transfer"
        ),
    )
    surface_detail_level: int = Field(
        ge=1, le=7,
        description=(
            "Is there sufficient surface detail for professional use "
            "(seams, edges, subtle texture variation)? "
            "1 = Far too low detail  7 = Professional-grade detail level"
        ),
    )

    # ── 2D → 3D fidelity ─────────────────────────────────────────────────────
    fidelity_2d_to_3d: int = Field(
        ge=1, le=7,
        description=(
            "How faithfully does the 3D scene reproduce the 2D variant image? "
            "1 = Looks like a different design  7 = Near-perfect match"
        ),
    )
    consistency_with_original_photo: int = Field(
        ge=1, le=7,
        description=(
            "Does the 3D reconstruction reflect the original room photograph's layout and structure? "
            "1 = Unrecognizable  7 = Clearly represents the same room"
        ),
    )
    furniture_identity_preserved: int = Field(
        ge=1, le=7,
        description=(
            "Are the same furniture pieces from the 2D variant faithfully reproduced in 3D? "
            "1 = Completely different objects  7 = Same pieces, accurately reconstructed"
        ),
    )
    palette_transfer_accuracy: int = Field(
        ge=1, le=7,
        description=(
            "How accurately was the 2D palette (colors, materials) transferred to the 3D textures? "
            "1 = No color relationship  7 = Pixel-accurate color match"
        ),
    )

    # ── Professional applicability ───────────────────────────────────────────
    usability_for_further_modeling: int = Field(
        ge=1, le=7,
        description=(
            "Could you use this 3D model as a starting point for further modeling work in your software of choice? "
            "1 = Would need to start from scratch  7 = Ready to use as-is"
        ),
    )
    bim_readiness: int = Field(
        ge=1, le=7,
        description=(
            "How ready is this output for use in a BIM workflow (ArchiCAD, Revit, IFC)? "
            "1 = Completely incompatible  7 = Import-ready for BIM"
        ),
    )
    presentation_quality: int = Field(
        ge=1, le=7,
        description=(
            "Is the 3D model of sufficient quality to render for client presentation? "
            "1 = Unusable  7 = Ready for high-quality client rendering"
        ),
    )
    overall_3d_quality: int = Field(
        ge=1, le=7,
        description="Overall holistic rating of the 3D output. 1 = Very poor  7 = Excellent",
    )

    # ── Categorical ──────────────────────────────────────────────────────────
    would_use_3d_output: Literal["yes_as_is", "yes_with_cleanup", "no"] = Field(
        ...,
        description=(
            "Would you use this 3D output in a real project? "
            "'yes_as_is' = use without modification, "
            "'yes_with_cleanup' = use after some fixing, "
            "'no' = would not use."
        ),
    )
    preferred_export_format: Literal["glb", "obj", "fbx", "ifc", "usdz", "3dm", "other"] = Field(
        ...,
        description="Which export format would be most useful for your workflow?",
    )
    preferred_export_format_other: Optional[str] = Field(
        default=None, description="If 'other', specify.", max_length=100
    )
    cleanup_effort_required: Literal["none", "minor", "moderate", "major", "complete_redo"] = Field(
        ...,
        description=(
            "How much cleanup work would be required before this 3D model is usable? "
            "'none' = usable immediately, 'minor' = < 30 min, 'moderate' = 1-4h, "
            "'major' = > 4h, 'complete_redo' = easier to rebuild from scratch."
        ),
    )

    # ── Open text ────────────────────────────────────────────────────────────
    mesh_issues_observed: Optional[str] = Field(
        default=None,
        description=(
            "Describe any mesh/geometry problems you observed "
            "(holes, wrong topology, incorrect object boundaries, missing surfaces, etc.)."
        ),
        max_length=1000,
    )
    texture_issues_observed: Optional[str] = Field(
        default=None,
        description=(
            "Describe any texture/material problems (blurry areas, color shifts, "
            "missing textures, tiling artifacts, etc.)."
        ),
        max_length=1000,
    )
    missing_elements: Optional[str] = Field(
        default=None,
        description=(
            "What elements are absent from the 3D model that should be present "
            "(e.g. specific furniture pieces, doors, windows, ceiling fixtures)?"
        ),
        max_length=1000,
    )
    unexpected_elements: Optional[str] = Field(
        default=None,
        description=(
            "What unexpected or incorrect elements appeared in the 3D model "
            "that were not in the original room or 2D variant?"
        ),
        max_length=1000,
    )
    additional_3d_comments: Optional[str] = Field(
        default=None,
        description=(
            "Any additional observations about the 3D output — technical, aesthetic, or practical. "
            "What would make you choose this over alternative reconstruction methods?"
        ),
        max_length=2000,
    )


class Commit3DEvalResponse(BaseModel):
    id: int
    study_session_id: int
    evaluator_profile_id: int
    committed_variant_id: int
    submitted_at: datetime

    model_config = {"from_attributes": True}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  RESEARCHER ENDPOINTS (JWT-protected)                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@router.post("/sessions", response_model=StudySessionResponse, status_code=201)
def create_study_session(
    body: CreateStudySessionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a study session and receive a participant-access token to share."""
    ss = evaluation_service.create_study_session(
        db=db,
        scene_id=body.scene_id,
        user_id=current_user.id,
        design_session_id=body.design_session_id,
        notes=body.notes,
        generation_engine_disclosed=body.generation_engine_disclosed,
        expires_at=body.expires_at,
    )
    return StudySessionResponse(
        id=ss.id,
        token=ss.token,
        scene_id=ss.scene_id,
        design_session_id=ss.design_session_id,
        generation_engine_disclosed=ss.generation_engine_disclosed,
        notes=ss.notes,
        created_at=ss.created_at,
        expires_at=ss.expires_at,
        participant_url_hint=f"/evaluation/{ss.token}/view",
    )


@router.get("/sessions", response_model=List[StudySessionResponse])
def list_study_sessions(
    scene_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    from app.repositories import evaluation_repo as repo
    sessions = repo.list_study_sessions(db, scene_id=scene_id)
    return [
        StudySessionResponse(
            id=ss.id,
            token=ss.token,
            scene_id=ss.scene_id,
            design_session_id=ss.design_session_id,
            generation_engine_disclosed=ss.generation_engine_disclosed,
            notes=ss.notes,
            created_at=ss.created_at,
            expires_at=ss.expires_at,
            participant_url_hint=f"/evaluation/{ss.token}/view",
        )
        for ss in sessions
    ]


@router.get("/export/profiles.csv", response_class=PlainTextResponse)
def export_profiles(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Download all evaluator demographics as CSV."""
    return PlainTextResponse(
        content=evaluation_service.export_profiles_csv(db),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=evaluator_profiles.csv"},
    )


@router.get("/export/variants_2d.csv", response_class=PlainTextResponse)
def export_variants_2d(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Download all per-variant 2D ratings as CSV."""
    return PlainTextResponse(
        content=evaluation_service.export_variant_csv(db),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=variant_2d_evaluations.csv"},
    )


@router.get("/export/variant_sets.csv", response_class=PlainTextResponse)
def export_variant_sets(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Download all cross-variant comparative evaluations as CSV."""
    return PlainTextResponse(
        content=evaluation_service.export_set_csv(db),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=variant_set_evaluations.csv"},
    )


@router.get("/export/commit_3d.csv", response_class=PlainTextResponse)
def export_commit_3d(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Download all 3D model evaluations as CSV."""
    return PlainTextResponse(
        content=evaluation_service.export_commit_csv(db),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=commit_3d_evaluations.csv"},
    )


@router.get("/export/all.csv", response_class=PlainTextResponse)
def export_all(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Download all evaluation tables concatenated into a single CSV file with section headers."""
    return PlainTextResponse(
        content=evaluation_service.export_csv_all(db),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=all_evaluations.csv"},
    )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PARTICIPANT ENDPOINTS (public — token is the credential)                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@router.get("/{token}/view", response_class=HTMLResponse)
def participant_viewer(token: str, db: Session = Depends(get_session)):
    """
    Landing page for study participants.
    Shows all generated room variants + 3D viewer, with the participant code UI.
    Share this URL (with the token) in your Google Forms invitation.
    """
    info = evaluation_service.get_session_info(db, token)
    variants_html = ""
    for i, v in enumerate(info["variant_images"]):
        label = chr(65 + i)  # A, B, C, D…
        img_url = f"/scenes/{info['scene_id']}/design/variants/{v['variant_id']}/image"
        variants_html += f"""
        <div class="variant-card">
          <div class="variant-label">Variant {label}</div>
          <img src="{img_url}" alt="Variant {label}" loading="lazy" />
          <div class="variant-meta">ID: {v['variant_id']}</div>
        </div>"""

    commit_section = ""
    if info["has_3d_commit"]:
        glb_url = f"/scenes/{info['scene_id']}/design/commit/download"
        commit_section = f"""
        <section class="section">
          <h2>3D Committed Scene</h2>
          <p>Use the viewer below to explore the 3D model. You can rotate, zoom, and pan.</p>
          <model-viewer
            src="{glb_url}"
            alt="3D room model"
            camera-controls auto-rotate
            style="width:100%;height:520px;background:#f0f0f0;border-radius:8px;">
          </model-viewer>
        </section>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Room Design Evaluation</title>
  <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.4.0/model-viewer.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Helvetica Neue', Arial, sans-serif; background: #fafafa; color: #1a1a1a; line-height: 1.6; }}
    .header {{ background: #1a1a1a; color: #fff; padding: 24px 40px; }}
    .header h1 {{ font-size: 1.5rem; font-weight: 400; letter-spacing: 0.05em; }}
    .header p {{ font-size: 0.9rem; opacity: 0.7; margin-top: 4px; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 40px; }}
    .section {{ margin-bottom: 48px; }}
    .section h2 {{ font-size: 1.1rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 16px; color: #555; border-bottom: 1px solid #e0e0e0; padding-bottom: 8px; }}
    .notice {{ background: #f5f5dc; border-left: 4px solid #c8a96e; padding: 16px 20px; border-radius: 4px; margin-bottom: 32px; }}
    .notice strong {{ display: block; margin-bottom: 4px; }}
    .variants-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 24px; }}
    .variant-card {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; }}
    .variant-label {{ background: #1a1a1a; color: #fff; padding: 10px 16px; font-size: 0.85rem; font-weight: 600; letter-spacing: 0.1em; }}
    .variant-card img {{ width: 100%; aspect-ratio: 4/3; object-fit: cover; display: block; }}
    .variant-meta {{ padding: 8px 16px; font-size: 0.75rem; color: #999; background: #fafafa; }}
    .instructions {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 24px; }}
    .instructions ol {{ padding-left: 20px; }}
    .instructions li {{ margin-bottom: 12px; font-size: 0.95rem; }}
    .instructions li strong {{ color: #1a1a1a; }}
    .step-number {{ display: inline-block; background: #1a1a1a; color: #fff; width: 24px; height: 24px; border-radius: 50%; text-align: center; line-height: 24px; font-size: 0.75rem; font-weight: 700; margin-right: 8px; }}
    footer {{ text-align: center; padding: 32px; font-size: 0.8rem; color: #aaa; border-top: 1px solid #e0e0e0; margin-top: 48px; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>AI-Assisted Interior Design — Participant Evaluation</h1>
    <p>Architecture Student Research Study &nbsp;|&nbsp; Session: {token[:8]}…</p>
  </div>
  <div class="container">
    <div class="notice">
      <strong>Before you begin:</strong>
      Keep this page open alongside the evaluation form. You will need to refer to the images and 3D model
      while answering questions. Your answers are recorded in the form — not on this page.
      Variant count: <strong>{info['variant_count']}</strong>.
    </div>

    <section class="section">
      <h2>Generated Room Variants</h2>
      <p style="margin-bottom:16px;font-size:0.9rem;color:#666;">
        Each variant below was generated from the same room photo and mood board, with different
        palette and furniture configurations. Study all variants carefully before answering the evaluation form.
      </p>
      <div class="variants-grid">
        {variants_html}
      </div>
    </section>

    {commit_section}

    <section class="section">
      <h2>How to complete the evaluation</h2>
      <div class="instructions">
        <ol>
          <li>
            <strong>Study all variants</strong> above. Take your time — zoom in if needed
            (right-click → Open image in new tab).
          </li>
          <li>
            <strong>Open the evaluation form</strong> shared with you by the researcher
            (Google Forms link in your invitation email).
          </li>
          <li>
            <strong>Enter your participant code</strong> — the code assigned to you by the researcher.
            You will need it in every section of the form.
          </li>
          <li>
            <strong>Fill in the demographics section</strong> first (background, experience, software use).
          </li>
          <li>
            <strong>Rate each variant individually</strong> using the per-variant section.
            The variant labels (A, B, C…) correspond to the images above, in order.
            Variant IDs shown in grey below each image are used in the ranking questions.
          </li>
          <li>
            <strong>Complete the comparative section</strong> after seeing all variants.
            This asks you to rank them and answer questions about how palette and layout varied.
          </li>
          {"<li><strong>Complete the 3D section</strong> after exploring the 3D viewer above.</li>" if info["has_3d_commit"] else ""}
        </ol>
      </div>
    </section>
  </div>
  <footer>Research study — Interior Design AI Evaluation &nbsp;|&nbsp; Do not share this link.</footer>
</body>
</html>"""
    return HTMLResponse(content=html)


@router.get("/{token}/info")
def session_info(token: str, db: Session = Depends(get_session)):
    """Public: variant metadata for this study session (no JWT required)."""
    return evaluation_service.get_session_info(db, token)


@router.post("/{token}/profile", response_model=EvaluatorProfileResponse, status_code=201)
def submit_profile(
    token: str,
    body: EvaluatorProfileForm,
    db: Session = Depends(get_db),
):
    """Submit the participant demographics form (Form 1 — fill once)."""
    data = body.model_dump()
    data["software_used_json"] = json.dumps(data.pop("software_used", []))
    return evaluation_service.submit_evaluator_profile(db, token, data)


@router.post("/{token}/variants/{variant_id}", response_model=Variant2DEvalResponse, status_code=201)
def submit_variant_evaluation(
    token: str,
    variant_id: int,
    body: Variant2DEvaluationForm,
    db: Session = Depends(get_db),
):
    """Submit a per-variant 2D rating (Form 2 — fill once per variant shown)."""
    data = body.model_dump()
    participant_code = data.pop("participant_code")
    return evaluation_service.submit_variant_evaluation(db, token, participant_code, variant_id, data)


@router.post("/{token}/set", response_model=VariantSetEvalResponse, status_code=201)
def submit_set_evaluation(
    token: str,
    body: VariantSetEvaluationForm,
    db: Session = Depends(get_db),
):
    """Submit the cross-variant comparative form (Form 3 — fill once after seeing all variants)."""
    data = body.model_dump()
    participant_code = data.pop("participant_code")
    data["variant_ranking_json"] = json.dumps(data.pop("variant_ranking", []))
    data["perceived_what_changed_json"] = json.dumps(data.pop("perceived_what_changed", {}))
    return evaluation_service.submit_set_evaluation(db, token, participant_code, data)


@router.post("/{token}/commit3d", response_model=Commit3DEvalResponse, status_code=201)
def submit_commit3d_evaluation(
    token: str,
    body: Commit3DEvaluationForm,
    db: Session = Depends(get_db),
):
    """Submit the 3D model evaluation (Form 4 — fill after exploring the 3D viewer)."""
    data = body.model_dump()
    participant_code = data.pop("participant_code")
    return evaluation_service.submit_commit3d_evaluation(db, token, participant_code, data)
