from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field


class StudySession(SQLModel, table=True):
    """A researcher-created token that grants participants access to a set of variants."""
    __tablename__ = "study_sessions"

    id: Optional[int] = Field(default=None, primary_key=True)
    token: str = Field(index=True, nullable=False)           # UUID shared with participants
    scene_id: int = Field(foreign_key="scenes.id", index=True, nullable=False)
    design_session_id: Optional[int] = Field(default=None, foreign_key="design_sessions.id")
    created_by_user_id: int = Field(foreign_key="users.id", nullable=False)
    presentation_order_json: str = Field(default="[]")       # ordered variant ID list
    generation_engine_disclosed: bool = Field(default=False) # was AI generation revealed?
    notes: Optional[str] = None                              # researcher notes
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


class EvaluatorProfile(SQLModel, table=True):
    """Demographic form — filled once per participant per study session."""
    __tablename__ = "evaluator_profiles"

    id: Optional[int] = Field(default=None, primary_key=True)
    study_session_id: int = Field(foreign_key="study_sessions.id", index=True, nullable=False)
    participant_code: str = Field(index=True)
    age_range: str                                           # "<25"|"25-34"|"35-44"|"45-54"|"55+"
    gender: Optional[str] = None                            # optional, self-described
    professional_role: str                                   # "student"|"architect"|"interior_designer"|"researcher"|"other"
    professional_role_other: Optional[str] = None
    years_experience: str                                    # "0-2"|"3-5"|"6-10"|"11-20"|">20"
    education_level: str                                     # "bachelors"|"masters"|"phd"|"professional_license"|"other"
    education_level_other: Optional[str] = None
    specialization: str                                      # "residential"|"commercial"|"hospitality"|"urban"|"multiple"|"other"
    specialization_other: Optional[str] = None
    country_of_education: str
    software_used_json: str = Field(default="[]")           # JSON list of tools
    familiarity_ai_tools: int                               # 1-5 Likert (1=never heard of, 5=use daily)
    familiarity_3d_modeling: int                            # 1-5
    familiarity_interior_design_software: int               # 1-5
    has_used_ai_design_tools_before: bool = Field(default=False)
    ai_tools_used_names: Optional[str] = None               # free text if above = true
    submitted_at: datetime = Field(default_factory=datetime.utcnow)


class Variant2DEvaluation(SQLModel, table=True):
    """Per-variant rating — one row per participant per variant."""
    __tablename__ = "variant_2d_evaluations"

    id: Optional[int] = Field(default=None, primary_key=True)
    study_session_id: int = Field(foreign_key="study_sessions.id", index=True, nullable=False)
    evaluator_profile_id: int = Field(foreign_key="evaluator_profiles.id", index=True, nullable=False)
    variant_id: int = Field(foreign_key="variants.id", index=True, nullable=False)
    presentation_position: int                              # 1..N — order shown (for position-bias analysis)
    engine_type: str = Field(default="diffusion")           # "diffusion"|"baseline" — ablation label

    # ── Appearance / palette channel ─────────────────────────────────────────
    aesthetic_quality: int                                  # 1-7
    color_harmony: int                                      # 1-7
    palette_fidelity_to_moodboard: int                      # 1-7: does color feel like the moodboard?
    atmosphere_mood: int                                    # 1-7: does it evoke the intended atmosphere?
    color_temperature_appropriateness: int                  # 1-7: warm/cool balance for the space

    # ── Structure / furniture channel ────────────────────────────────────────
    spatial_layout_preservation: int                        # 1-7: are walls/windows/doors preserved?
    furniture_style_consistency: int                        # 1-7: do pieces feel stylistically coherent?
    furniture_placement_plausibility: int                   # 1-7: are objects physically plausible?
    scale_and_proportion_accuracy: int                      # 1-7: do proportions match real rooms?

    # ── Realism / quality ────────────────────────────────────────────────────
    photorealism: int                                       # 1-7
    lighting_plausibility: int                              # 1-7: does lighting feel physically correct?
    material_surface_quality: int                           # 1-7: textures/finishes look believable?
    shadow_and_reflection_quality: int                      # 1-7

    # ── Professional applicability ───────────────────────────────────────────
    professional_suitability: int                           # 1-7: suitable for professional use?
    client_presentability: int                              # 1-7: would you show this to a client?
    innovation_and_creativity: int                          # 1-7: is the design novel/interesting?
    design_coherence: int                                   # 1-7: does it feel like a unified design?

    # ── Binary / categorical ─────────────────────────────────────────────────
    would_present_to_client: str                            # "yes"|"no"|"maybe"
    manual_redesign_time_estimate: str                      # "<1h"|"1-4h"|"4-8h"|"1-3d"|">3d"

    # ── Open text ────────────────────────────────────────────────────────────
    most_appealing_aspect: Optional[str] = None
    most_problematic_aspect: Optional[str] = None
    design_suggestions: Optional[str] = None               # specific design improvements they'd make
    elements_that_look_ai_generated: Optional[str] = None  # what reveals the AI origin?

    submitted_at: datetime = Field(default_factory=datetime.utcnow)


class VariantSetEvaluation(SQLModel, table=True):
    """Cross-variant comparison — filled once after seeing all variants."""
    __tablename__ = "variant_set_evaluations"

    id: Optional[int] = Field(default=None, primary_key=True)
    study_session_id: int = Field(foreign_key="study_sessions.id", index=True, nullable=False)
    evaluator_profile_id: int = Field(foreign_key="evaluator_profiles.id", index=True, nullable=False)
    scene_id: int = Field(foreign_key="scenes.id", index=True, nullable=False)

    # ── Ranking ──────────────────────────────────────────────────────────────
    variant_ranking_json: str = Field(default="[]")        # variant IDs ordered best→worst
    preferred_variant_id: Optional[int] = Field(default=None, foreign_key="variants.id")
    preferred_variant_reason: Optional[str] = None

    # ── Disentanglement perception (core research probe) ─────────────────────
    palette_change_perceived: bool                          # noticed color palette changed across variants?
    furniture_change_perceived: bool                        # noticed furniture layout changed?
    perceived_what_changed_json: str = Field(default="{}")
    # ^ {"palette": bool, "furniture": bool, "both": bool, "neither": bool} for each variant pair
    disentanglement_clarity: int                            # 1-7: how clearly were the two axes separated?
    palette_axis_control_confidence: int                    # 1-7: did palette changes feel intentional/controlled?
    furniture_axis_control_confidence: int                  # 1-7: did furniture changes feel intentional?
    cross_channel_leakage_observed: bool                    # did palette changes alter furniture or vice versa?
    leakage_description: Optional[str] = None              # if leakage observed: describe what

    # ── Comparative quality ───────────────────────────────────────────────────
    consistency_across_variants: int                        # 1-7: do all variants feel like the same room?
    variation_diversity: int                                # 1-7: are variants sufficiently different?
    best_palette_match_variant_id: Optional[int] = Field(default=None, foreign_key="variants.id")
    best_layout_match_variant_id: Optional[int] = Field(default=None, foreign_key="variants.id")

    # ── AI tool assessment ────────────────────────────────────────────────────
    overall_ai_tool_utility: int                            # 1-7: useful for professional workflow?
    overall_ai_tool_trustworthiness: int                    # 1-7: can you rely on the output?
    workflow_integration_ease: int                          # 1-7: easy to fit into your design process?
    time_saved_vs_manual: str                               # "none"|"<30min"|"1-2h"|"3-4h"|">4h"
    would_use_professionally: str                           # "yes"|"no"|"maybe"|"already_do"
    tool_replaces_or_augments: str                          # "replaces_sketch"|"augments_sketch"|"replaces_rendering"|"augments_rendering"|"other"
    tool_replaces_or_augments_other: Optional[str] = None

    # ── Open text ─────────────────────────────────────────────────────────────
    biggest_limitation: Optional[str] = None
    most_valuable_feature: Optional[str] = None
    suggested_improvements: Optional[str] = None
    comparison_to_existing_tools: Optional[str] = None     # how does this compare to SketchUp, Revit, etc.?
    additional_comments: Optional[str] = None

    submitted_at: datetime = Field(default_factory=datetime.utcnow)


class Commit3DEvaluation(SQLModel, table=True):
    """Evaluation of the 3D committed GLB — filled after viewing the 3D model."""
    __tablename__ = "commit_3d_evaluations"

    id: Optional[int] = Field(default=None, primary_key=True)
    study_session_id: int = Field(foreign_key="study_sessions.id", index=True, nullable=False)
    evaluator_profile_id: int = Field(foreign_key="evaluator_profiles.id", index=True, nullable=False)
    scene_id: int = Field(foreign_key="scenes.id", index=True, nullable=False)
    committed_variant_id: int = Field(foreign_key="variants.id", index=True, nullable=False)

    # ── Geometry accuracy ────────────────────────────────────────────────────
    mesh_completeness: int                                  # 1-7: are all room surfaces present?
    spatial_accuracy_dimensions: int                        # 1-7: do room proportions feel right?
    furniture_geometry_accuracy: int                        # 1-7: do objects have correct 3D shape?
    room_proportions_accuracy: int                          # 1-7: height/width/depth feel plausible?
    artifact_presence: int                                  # 1-7: 7=no artifacts, 1=severe artifacts

    # ── Appearance / texture ─────────────────────────────────────────────────
    texture_quality: int                                    # 1-7
    material_representation: int                            # 1-7: materials look like real surfaces?
    palette_fidelity_3d: int                                # 1-7: does 3D color match the chosen variant?
    surface_detail_level: int                               # 1-7: enough detail for professional use?

    # ── 2D → 3D fidelity ─────────────────────────────────────────────────────
    fidelity_2d_to_3d: int                                  # 1-7: does 3D match the 2D variant?
    consistency_with_original_photo: int                    # 1-7: does it reflect the original room?
    furniture_identity_preserved: int                       # 1-7: same furniture pieces as 2D?
    palette_transfer_accuracy: int                          # 1-7: did palette translate from 2D to 3D?

    # ── Professional applicability ───────────────────────────────────────────
    usability_for_further_modeling: int                     # 1-7: could you keep modeling from this?
    bim_readiness: int                                      # 1-7: usable in BIM/ArchiCAD/Revit?
    presentation_quality: int                               # 1-7: usable for client presentations?
    overall_3d_quality: int                                 # 1-7: holistic score

    # ── Categorical ─────────────────────────────────────────────────────────
    would_use_3d_output: str                                # "yes_as_is"|"yes_with_cleanup"|"no"
    preferred_export_format: str                            # "glb"|"obj"|"fbx"|"ifc"|"other"
    preferred_export_format_other: Optional[str] = None
    cleanup_effort_required: str                            # "none"|"minor"|"moderate"|"major"|"complete_redo"

    # ── Open text ────────────────────────────────────────────────────────────
    mesh_issues_observed: Optional[str] = None
    texture_issues_observed: Optional[str] = None
    missing_elements: Optional[str] = None                  # what's absent from the 3D that should be there?
    unexpected_elements: Optional[str] = None               # what appeared that shouldn't have?
    additional_3d_comments: Optional[str] = None

    submitted_at: datetime = Field(default_factory=datetime.utcnow)
