/**
 * Google Apps Script — AI Interior Design Study Evaluation Form
 *
 * HOW TO USE:
 *   1. Go to https://script.google.com and create a new project.
 *   2. Paste this entire file into the editor (replace the default content).
 *   3. Click Run → createEvaluationForm.
 *   4. Grant the permissions Google requests (to create Forms and Docs on your account).
 *   5. The script logs the form URL when done. Open it to review and share.
 *
 * The script creates a single Google Form with 7 sections:
 *   Section 0 — Instructions (intro page)
 *   Section 1 — Participant background (demographics)
 *   Section 2 — Variant A rating
 *   Section 3 — Variant B rating
 *   Section 4 — Variant C rating
 *   Section 5 — Variant D rating
 *   Section 6 — Cross-variant comparison (disentanglement probe)
 *   Section 7 — 3D model evaluation
 *
 * Adjust STUDY_TITLE, VIEWER_BASE_URL, and VARIANT_COUNT before running.
 */

// ── Configuration ─────────────────────────────────────────────────────────────

var STUDY_TITLE       = "AI-Assisted Interior Design — Architecture Student Evaluation";
var STUDY_DESCRIPTION = "Thank you for participating in this research study. " +
  "Your responses will be used to evaluate an AI system that generates interior " +
  "design iterations from a room photo and a mood board. The study takes approximately " +
  "25–40 minutes. All responses are anonymous.";

// Replace with the actual URL the researcher shared, e.g. https://your-server/evaluation/abc123/view
var VIEWER_BASE_URL   = "REPLACE_WITH_YOUR_VIEWER_URL";

// Number of generated variants (typically 4). Change if fewer were generated.
var VARIANT_COUNT     = 4;

// Set to false to omit the 3D evaluation section (if no 3D commit was generated).
var INCLUDE_3D        = true;


// ── Helper builders ───────────────────────────────────────────────────────────

function scale7(form, title, helpText, lowLabel, highLabel) {
  lowLabel  = lowLabel  || "1 — Strongly disagree / Very poor";
  highLabel = highLabel || "7 — Strongly agree / Excellent";
  return form.addScaleItem()
    .setTitle(title)
    .setHelpText(helpText || "")
    .setBounds(1, 7)
    .setLabels(lowLabel, highLabel)
    .setRequired(true);
}

function scale5(form, title, helpText, lowLabel, highLabel) {
  lowLabel  = lowLabel  || "1 — No experience / Never";
  highLabel = highLabel || "5 — Expert / Daily use";
  return form.addScaleItem()
    .setTitle(title)
    .setHelpText(helpText || "")
    .setBounds(1, 5)
    .setLabels(lowLabel, highLabel)
    .setRequired(true);
}

function radio(form, title, choices, helpText, required) {
  required = (required === undefined) ? true : required;
  var item = form.addMultipleChoiceItem()
    .setTitle(title)
    .setChoiceValues(choices)
    .setRequired(required);
  if (helpText) item.setHelpText(helpText);
  return item;
}

function checkbox(form, title, choices, helpText, required) {
  required = (required === undefined) ? false : required;
  var item = form.addCheckboxItem()
    .setTitle(title)
    .setChoiceValues(choices)
    .setRequired(required);
  if (helpText) item.setHelpText(helpText);
  return item;
}

function shortText(form, title, helpText, required) {
  required = (required === undefined) ? true : required;
  var item = form.addTextItem().setTitle(title).setRequired(required);
  if (helpText) item.setHelpText(helpText);
  return item;
}

function longText(form, title, helpText, required) {
  required = (required === undefined) ? false : required;
  var item = form.addParagraphTextItem().setTitle(title).setRequired(required);
  if (helpText) item.setHelpText(helpText);
  return item;
}

function pageBreak(form, title, description) {
  var item = form.addPageBreakItem().setTitle(title);
  if (description) item.setHelpText(description);
  return item;
}

function sectionHeader(form, title, description) {
  var item = form.addSectionHeaderItem().setTitle(title);
  if (description) item.setHelpText(description);
  return item;
}


// ── Main ──────────────────────────────────────────────────────────────────────

function createEvaluationForm() {
  var form = FormApp.create(STUDY_TITLE);
  form.setDescription(STUDY_DESCRIPTION);
  form.setCollectEmail(false);
  form.setAllowResponseEdits(false);
  form.setProgressBar(true);
  form.setShuffleQuestions(false);


  // ══════════════════════════════════════════════════════════════════════════
  // SECTION 0 — Instructions
  // ══════════════════════════════════════════════════════════════════════════

  sectionHeader(form, "Before you begin",
    "This form has 7 sections. Please read the instructions below carefully before starting.\n\n" +
    "IMPORTANT: Keep the variant viewer page open in a separate tab throughout the study.\n" +
    "You need to look at the images while answering questions.\n\n" +
    "Viewer URL: " + VIEWER_BASE_URL + "\n\n" +
    "Do not close the viewer tab. The images are labeled Variant A, B, C" + (VARIANT_COUNT > 3 ? ", D" : "") + " in order."
  );

  shortText(form, "Participant code",
    "Enter the code assigned to you by the researcher (e.g. P001). " +
    "You will need this code in every section — do not lose it.",
    true
  );


  // ══════════════════════════════════════════════════════════════════════════
  // SECTION 1 — Participant background (demographics)
  // ══════════════════════════════════════════════════════════════════════════

  pageBreak(form, "Section 1 of 7 — Your background",
    "This section collects anonymised demographic information about your professional " +
    "experience and familiarity with relevant tools. All fields are required unless marked optional."
  );

  radio(form, "Age range", ["Under 25", "25–34", "35–44", "45–54", "55 or older"]);

  shortText(form, "Gender (optional — self-described)",
    "Leave blank if you prefer not to answer.", false
  );

  radio(form, "Primary professional role", [
    "Architecture student (undergraduate)",
    "Architecture student (postgraduate / MSc / MArch)",
    "Practising architect",
    "Interior designer",
    "Academic / researcher",
    "Other"
  ]);

  shortText(form, "If 'Other' for professional role — please specify", "", false);

  radio(form, "Years of professional experience in architecture or design", [
    "0–2 years",
    "3–5 years",
    "6–10 years",
    "11–20 years",
    "More than 20 years"
  ]);

  radio(form, "Highest completed level of education", [
    "Bachelor's degree",
    "Master's degree",
    "PhD / Doctorate",
    "Professional licence (ARB / RIBA / AIA equivalent)",
    "Other"
  ]);

  shortText(form, "If 'Other' for education — please specify", "", false);

  radio(form, "Primary design specialisation", [
    "Residential",
    "Commercial / office",
    "Hospitality / retail",
    "Urban planning / masterplanning",
    "Landscape architecture",
    "Heritage conservation",
    "Multiple / generalist",
    "Other"
  ]);

  shortText(form, "If 'Other' for specialisation — please specify", "", false);

  shortText(form, "Country where you completed your primary design education",
    "E.g. Romania, United Kingdom, Germany."
  );

  checkbox(form, "Design and visualisation software you use regularly (select all that apply)", [
    "AutoCAD",
    "Revit",
    "ArchiCAD",
    "SketchUp",
    "Rhino / Grasshopper",
    "Blender",
    "3ds Max",
    "Lumion",
    "Enscape",
    "V-Ray",
    "Vectorworks",
    "Adobe Photoshop / Illustrator",
    "Adobe Firefly / Generative Fill",
    "Midjourney",
    "DALL-E / ChatGPT image generation",
    "Stable Diffusion",
    "Planner5D / Roomstyler",
    "Other"
  ],
  "Select every tool you use at least occasionally in your professional or academic practice.",
  false
  );

  shortText(form, "If 'Other' software — please list them", "", false);

  scale5(form, "How familiar are you with AI-powered design tools?",
    "1 = I have never heard of them\n" +
    "2 = I have heard of them but never used any\n" +
    "3 = I have tried them occasionally\n" +
    "4 = I use them regularly (at least weekly)\n" +
    "5 = I use them daily / I consider myself an expert user",
    "1 — Never heard of them",
    "5 — Expert / daily use"
  );

  scale5(form, "How proficient are you with 3D modelling software?",
    "1 = No experience at all\n2 = Beginner\n3 = Intermediate\n4 = Advanced\n5 = Expert",
    "1 — No experience",
    "5 — Expert"
  );

  scale5(form, "How proficient are you with interior-design-specific software?",
    "E.g. Planner5D, Roomstyler, Homestyler, DesignApp.\n" +
    "1 = No experience  2 = Beginner  3 = Intermediate  4 = Advanced  5 = Expert",
    "1 — No experience",
    "5 — Expert"
  );

  radio(form, "Have you used any AI-assisted design or image generation tool before this study?", [
    "Yes",
    "No"
  ]);

  shortText(form, "If yes — which AI design tools have you used before?",
    "E.g. Midjourney, DALL-E, Adobe Firefly, Stable Diffusion, Dall-E in Revit, etc. " +
    "Leave blank if no.",
    false
  );


  // ══════════════════════════════════════════════════════════════════════════
  // SECTIONS 2–5 — Per-variant 2D evaluations
  // ══════════════════════════════════════════════════════════════════════════

  var variantLabels = ["A", "B", "C", "D", "E", "F"];

  for (var vi = 0; vi < VARIANT_COUNT; vi++) {
    var label = variantLabels[vi];
    var sectionNum = vi + 2;

    pageBreak(form,
      "Section " + sectionNum + " of " + (VARIANT_COUNT + 3) + " — Rate Variant " + label,
      "Open the viewer page and look at Variant " + label + " carefully before answering.\n" +
      "All rating scales are 1 to 7 unless otherwise noted.\n\n" +
      "Viewer: " + VIEWER_BASE_URL
    );

    sectionHeader(form, "Variant " + label + " — Appearance and colour palette",
      "These questions assess the colour and atmospheric qualities of Variant " + label + "."
    );

    scale7(form,
      label + ". Aesthetic quality",
      "Overall aesthetic quality of the room design.",
      "1 — Very poor", "7 — Excellent"
    );
    scale7(form,
      label + ". Color harmony",
      "How harmoniously do the colours in this room work together?",
      "1 — Clashing / discordant", "7 — Perfectly harmonious"
    );
    scale7(form,
      label + ". Palette fidelity to the mood board",
      "How well does the room's colour palette reflect the mood board you were shown before the study? " +
      "1 = No resemblance to the mood board  7 = Perfectly faithful to the mood board.",
      "1 — No resemblance to mood board", "7 — Perfectly faithful"
    );
    scale7(form,
      label + ". Atmosphere and mood",
      "How successfully does this design evoke the atmosphere or mood intended by the mood board?",
      "1 — Not at all", "7 — Exactly as intended"
    );
    scale7(form,
      label + ". Colour temperature appropriateness",
      "Is the warm/cool balance of the colour palette appropriate for this type of interior space?",
      "1 — Very inappropriate", "7 — Perfectly appropriate"
    );

    sectionHeader(form, "Variant " + label + " — Spatial layout and furniture",
      "These questions assess how well the original room's structure and furniture are preserved."
    );

    scale7(form,
      label + ". Spatial layout preservation",
      "How well is the original room's spatial layout preserved — walls, windows, doors, openings? " +
      "1 = Completely changed  7 = Identical layout.",
      "1 — Completely changed", "7 — Identical layout"
    );
    scale7(form,
      label + ". Furniture style consistency",
      "Do the furniture pieces feel stylistically coherent with each other and the overall room?",
      "1 — Completely inconsistent", "7 — Perfectly coherent"
    );
    scale7(form,
      label + ". Furniture placement plausibility",
      "Are the furniture pieces placed in physically and functionally plausible positions?",
      "1 — Impossible / bizarre placements", "7 — Natural, functional placement"
    );
    scale7(form,
      label + ". Scale and proportion accuracy",
      "Do the furniture pieces appear correctly scaled relative to the room and to each other?",
      "1 — Severely disproportionate", "7 — Accurately proportioned"
    );

    sectionHeader(form, "Variant " + label + " — Realism and rendering quality");

    scale7(form,
      label + ". Photorealism",
      "How photorealistic does the image look overall? " +
      "1 = Clearly computer-generated / cartoonish  7 = Indistinguishable from a real photograph.",
      "1 — Clearly artificial", "7 — Indistinguishable from a photo"
    );
    scale7(form,
      label + ". Lighting plausibility",
      "How physically plausible is the lighting in the scene?",
      "1 — Completely unrealistic", "7 — Fully convincing lighting"
    );
    scale7(form,
      label + ". Material and surface quality",
      "How convincing are the material textures and surface finishes (wood grain, fabric, tile, etc.)?",
      "1 — Fake / blurry textures", "7 — Convincing, high-quality surfaces"
    );
    scale7(form,
      label + ". Shadow and reflection quality",
      "How realistic are the shadows, reflections, and ambient occlusion?",
      "1 — Absent or wrong", "7 — Physically accurate"
    );

    sectionHeader(form, "Variant " + label + " — Professional applicability");

    scale7(form,
      label + ". Professional suitability",
      "How suitable is this output for professional use in an architecture or interior design practice?",
      "1 — Unusable professionally", "7 — Ready for direct professional use"
    );
    scale7(form,
      label + ". Client presentability",
      "Would you be confident showing this image to a client as a design proposal?",
      "1 — Definitely not", "7 — Absolutely, without modification"
    );
    scale7(form,
      label + ". Innovation and creativity",
      "How novel or creatively interesting is the design solution?",
      "1 — Generic / uninspired", "7 — Highly innovative"
    );
    scale7(form,
      label + ". Design coherence",
      "Does the overall design feel like a unified, intentional concept?",
      "1 — Random / incoherent", "7 — Fully unified design concept"
    );

    sectionHeader(form, "Variant " + label + " — Overall assessment");

    radio(form,
      label + ". Would you present this variant to a real client as a design option?",
      ["Yes", "No", "Maybe / with modifications"]
    );

    radio(form,
      label + ". How long would it take you to produce an equivalent result manually?",
      ["Less than 1 hour", "1–4 hours", "4–8 hours", "1–3 days", "More than 3 days"],
      "Estimate for one design iteration cycle, not the final deliverable."
    );

    longText(form,
      label + ". What is the single most appealing aspect of this design?",
      "Be specific — e.g. 'the warm terracotta tones contrast beautifully with the light oak flooring'."
    );

    longText(form,
      label + ". What is the most significant problem or weakness in this variant?",
      "Be as specific as possible."
    );

    longText(form,
      label + ". As a design professional, what specific changes would you recommend?",
      "Consider: colour, materials, furniture selection, scale, layout, lighting, style coherence. " +
      "Prioritise the most impactful changes."
    );

    longText(form,
      label + ". Which elements, if any, make it apparent this image was AI-generated?",
      "E.g. 'the upholstery texture looks unrealistically uniform', 'window reflections are wrong'. " +
      "Leave blank if nothing reveals an AI origin."
    );
  }


  // ══════════════════════════════════════════════════════════════════════════
  // SECTION 6 — Cross-variant comparison (disentanglement probe)
  // ══════════════════════════════════════════════════════════════════════════

  var sec6 = VARIANT_COUNT + 2;
  var totalSections = VARIANT_COUNT + 3 + (INCLUDE_3D ? 1 : 0);

  pageBreak(form,
    "Section " + sec6 + " of " + totalSections + " — Comparing all variants",
    "Complete this section AFTER you have rated all individual variants.\n" +
    "This section is the core of the study — it asks you to compare variants and assess " +
    "how independently the colour palette and furniture were controlled.\n\n" +
    "Viewer: " + VIEWER_BASE_URL
  );

  // Ranking
  sectionHeader(form, "Variant ranking",
    "Rank the variants from best to worst."
  );

  radio(form, "Which variant is your FIRST choice (best overall)?",
    variantLabels.slice(0, VARIANT_COUNT).map(function(l) { return "Variant " + l; })
  );
  radio(form, "Which variant is your SECOND choice?",
    variantLabels.slice(0, VARIANT_COUNT).map(function(l) { return "Variant " + l; })
  );
  if (VARIANT_COUNT >= 3) {
    radio(form, "Which variant is your THIRD choice?",
      variantLabels.slice(0, VARIANT_COUNT).map(function(l) { return "Variant " + l; })
    );
  }
  if (VARIANT_COUNT >= 4) {
    radio(form, "Which variant is your FOURTH choice (least preferred)?",
      variantLabels.slice(0, VARIANT_COUNT).map(function(l) { return "Variant " + l; })
    );
  }

  longText(form, "Why is your first-choice variant your preferred option?",
    "Explain in terms of palette, layout, atmosphere, or professional suitability. " +
    "What specifically makes it stand out from the others?"
  );

  // Disentanglement perception probe
  sectionHeader(form, "Disentanglement perception",
    "These questions are the core of the research. They assess whether you perceived " +
    "the colour palette and the furniture as independently varied across the variants."
  );

  radio(form,
    "Looking across all variants: did you perceive the COLOUR PALETTE changing between variants?",
    ["Yes, clearly", "Yes, somewhat", "Not sure", "No"],
    "Colour palette = the dominant wall colours, upholstery tones, accent colours."
  );

  radio(form,
    "Did you perceive the FURNITURE LAYOUT or FURNITURE STYLE changing between variants?",
    ["Yes, clearly", "Yes, somewhat", "Not sure", "No"],
    "Furniture = the pieces present, their style, placement, and the spatial arrangement."
  );

  checkbox(form, "Which of the following did you notice changing across the variants? (select all that apply)",
    [
      "Overall colour palette",
      "Wall colour / paint",
      "Upholstery and soft furnishing colour",
      "Flooring colour or material",
      "Accent colours and accessories",
      "Furniture piece identity (different objects)",
      "Furniture placement / arrangement",
      "Furniture style (e.g. Scandinavian vs industrial)",
      "Lighting warmth or intensity",
      "Material surfaces (e.g. wood vs fabric)",
      "Spatial layout (walls, windows, doors)",
      "Overall atmosphere",
      "Nothing — all variants looked the same to me"
    ],
    "Select everything you noticed varying, even if subtly."
  );

  scale7(form,
    "How clearly were the colour palette and furniture kept as SEPARATE, independently varied axes?",
    "Did the palette and furniture seem like two knobs you could turn independently, " +
    "or did they always change together?\n" +
    "1 = They seemed to change together / completely inseparable\n" +
    "4 = Somewhat separable — I could see two distinct axes but they overlapped\n" +
    "7 = Clearly and independently controllable — changing one left the other unchanged",
    "1 — Always changed together", "7 — Clearly independent"
  );

  scale7(form,
    "How confident are you that COLOUR PALETTE changes were deliberate and controlled (not random)?",
    "1 = Looks completely random  7 = Clearly intentional and consistent",
    "1 — Looks random", "7 — Clearly intentional"
  );

  scale7(form,
    "How confident are you that FURNITURE changes were deliberate and controlled (not random)?",
    "1 = Looks completely random  7 = Clearly intentional and consistent",
    "1 — Looks random", "7 — Clearly intentional"
  );

  radio(form,
    "Did you observe 'leakage' — where a change intended for one axis unexpectedly altered the other?",
    [
      "Yes — a colour change also altered furniture shape or placement",
      "Yes — a furniture change also altered the colour palette",
      "Yes — both types of leakage occurred",
      "No — the two axes appeared clean and independent",
      "I could not tell"
    ],
    "Leakage = a change in colour unexpectedly moved furniture, or vice versa."
  );

  longText(form,
    "If you observed leakage — describe it specifically",
    "Which variants were affected? What changed that shouldn't have? Leave blank if no leakage."
  );

  // Comparative quality
  sectionHeader(form, "Comparative quality across variants");

  scale7(form,
    "Do all variants feel like they belong to the same underlying room?",
    "1 = Completely different rooms — I would not recognise them as the same space\n" +
    "7 = Clearly the same room with controlled variations",
    "1 — Completely different rooms", "7 — Clearly the same room"
  );

  scale7(form,
    "Are the variants sufficiently different from each other to be useful as design alternatives?",
    "1 = All look identical — no real choice  7 = Meaningfully distinct alternatives",
    "1 — All look identical", "7 — Meaningfully distinct"
  );

  radio(form, "Which variant best matched the mood board colour palette?",
    variantLabels.slice(0, VARIANT_COUNT).map(function(l) { return "Variant " + l; }).concat(["None / tie"])
  );

  radio(form, "Which variant best preserved the original room's spatial layout?",
    variantLabels.slice(0, VARIANT_COUNT).map(function(l) { return "Variant " + l; }).concat(["None / tie"])
  );

  // AI tool assessment
  sectionHeader(form, "AI tool assessment",
    "These questions assess the potential of this kind of AI-assisted design tool " +
    "for professional architectural and interior design practice."
  );

  scale7(form,
    "How useful would this AI design tool be in your professional design workflow?",
    "1 = Not useful at all — would never use it\n" +
    "7 = Extremely useful — would integrate it into every relevant project",
    "1 — Not useful at all", "7 — Extremely useful"
  );

  scale7(form,
    "How much would you trust AI-generated outputs like these in a real project?",
    "1 = Not at all — would never rely on it without complete redesign\n" +
    "7 = Fully trust — would rely on it directly without checking",
    "1 — Not at all", "7 — Fully trust"
  );

  scale7(form,
    "How easy would it be to integrate this kind of tool into your existing design workflow?",
    "1 = Would require completely redesigning my workflow\n" +
    "7 = Fits seamlessly into my existing practice",
    "1 — Major workflow change required", "7 — Fits seamlessly"
  );

  radio(form,
    "Compared to producing equivalent results manually, how much time would this tool save per design iteration?",
    ["No time saved", "Less than 30 minutes", "1–2 hours", "3–4 hours", "More than 4 hours"]
  );

  radio(form,
    "Would you use a tool like this in your professional practice?",
    ["Yes — I would adopt it today", "Yes — once it improves", "Maybe", "No", "I already use similar tools"]
  );

  radio(form,
    "Which part of your workflow does this tool best fit?",
    [
      "Replaces hand sketching / early ideation",
      "Augments hand sketching / early ideation",
      "Replaces digital rendering / visualisation",
      "Augments digital rendering / visualisation",
      "Replaces client presentation preparation",
      "Augments client presentation preparation",
      "A completely different part of the workflow",
      "Does not fit into my workflow"
    ]
  );

  shortText(form, "If 'A completely different part' — describe which part", "", false);

  // Open text
  sectionHeader(form, "Open-ended feedback (cross-variant)");

  longText(form,
    "What is the single biggest limitation of this AI design approach for real professional use?",
    "Consider: output quality, controllability, reliability, client communication, " +
    "integration with BIM, or legal/copyright concerns."
  );

  longText(form,
    "What is the most valuable feature demonstrated by this system?",
    "What would make it worth adopting despite its limitations?"
  );

  longText(form,
    "What improvements would make this tool significantly more useful?",
    "Prioritise by impact — what ONE change would matter most to you as a practitioner?"
  );

  longText(form,
    "How does this AI approach compare to visualisation tools you currently use?",
    "E.g. compared to Lumion, Enscape, V-Ray, or using Midjourney for mood boards — " +
    "what does it do better or worse?"
  );

  longText(form,
    "Any other observations, reactions, or feedback not covered by the questions above?", ""
  );


  // ══════════════════════════════════════════════════════════════════════════
  // SECTION 7 — 3D model evaluation (optional)
  // ══════════════════════════════════════════════════════════════════════════

  if (INCLUDE_3D) {
    pageBreak(form,
      "Section " + totalSections + " of " + totalSections + " — 3D model evaluation",
      "Complete this section AFTER exploring the 3D model in the viewer on the participant page.\n" +
      "Use your mouse to rotate, zoom, and pan. Inspect the geometry and textures from multiple angles.\n\n" +
      "Viewer: " + VIEWER_BASE_URL
    );

    shortText(form, "Which variant was committed to 3D? (enter the letter — A, B, C or D)",
      "The researcher should have indicated this in the study invitation. " +
      "Check the viewer page if unsure.",
      true
    );

    // Geometry
    sectionHeader(form, "3D geometry accuracy");

    scale7(form,
      "Mesh completeness",
      "Are all expected room surfaces present and closed — floor, ceiling, walls, furniture?",
      "1 — Many surfaces missing / open holes", "7 — Fully complete mesh"
    );
    scale7(form,
      "Spatial accuracy and room dimensions",
      "Do the room's overall dimensions and proportions feel physically accurate?",
      "1 — Severely distorted", "7 — Accurately proportioned"
    );
    scale7(form,
      "Furniture geometry accuracy",
      "Do individual furniture objects have a correct 3D shape that matches their real-world counterpart?",
      "1 — Unrecognisable shapes", "7 — Accurately modelled"
    );
    scale7(form,
      "Room proportions (ceiling height, width, depth)",
      "Does the spatial envelope feel like a real room — not too tall, cramped, or wide?",
      "1 — Completely wrong proportions", "7 — Feels like a real room"
    );
    scale7(form,
      "Absence of mesh artefacts",
      "Rate the ABSENCE of artefacts: floating polygons, holes, intersecting geometry, z-fighting.\n" +
      "7 = No visible artefacts  1 = Severe, distracting artefacts",
      "1 — Severe artefacts", "7 — No artefacts"
    );

    // Appearance
    sectionHeader(form, "3D appearance and textures");

    scale7(form,
      "Texture quality",
      "Overall quality of the surface textures applied to the 3D model.",
      "1 — Blurry / missing textures", "7 — Sharp, high-quality textures"
    );
    scale7(form,
      "Material differentiation",
      "Do material properties look realistically differentiated — e.g. wood roughness vs fabric softness vs tile gloss?",
      "1 — All surfaces look identical", "7 — Convincingly differentiated materials"
    );
    scale7(form,
      "Colour palette fidelity in 3D",
      "How well does the 3D model's colour palette match the 2D variant it was generated from?",
      "1 — Colours completely changed", "7 — Faithful colour transfer"
    );
    scale7(form,
      "Surface detail level",
      "Is there sufficient surface detail for professional use — seams, edges, subtle texture variation?",
      "1 — Far too low detail", "7 — Professional-grade detail level"
    );

    // Fidelity
    sectionHeader(form, "2D → 3D fidelity");

    scale7(form,
      "Fidelity from 2D variant to 3D scene",
      "How faithfully does the 3D scene reproduce the 2D variant image you rated earlier?",
      "1 — Looks like a different design", "7 — Near-perfect match"
    );
    scale7(form,
      "Consistency with the original room photograph",
      "Does the 3D reconstruction reflect the original room photograph's layout and structure?",
      "1 — Unrecognisable as the same room", "7 — Clearly represents the same room"
    );
    scale7(form,
      "Furniture identity preserved in 3D",
      "Are the same furniture pieces from the 2D variant faithfully reproduced in 3D?",
      "1 — Completely different objects", "7 — Same pieces, accurately reconstructed"
    );
    scale7(form,
      "Colour palette transfer accuracy (2D to 3D)",
      "How accurately was the 2D colour palette translated into the 3D textures?",
      "1 — No colour relationship", "7 — Pixel-accurate colour match"
    );

    // Professional
    sectionHeader(form, "Professional applicability of the 3D output");

    scale7(form,
      "Usability for further modelling",
      "Could you use this 3D model as a starting point for further work in your software of choice?",
      "1 — Would need to start from scratch", "7 — Ready to use as-is"
    );
    scale7(form,
      "BIM readiness",
      "How ready is this output for use in a BIM workflow (ArchiCAD, Revit, IFC export)?",
      "1 — Completely incompatible with BIM", "7 — Import-ready for BIM"
    );
    scale7(form,
      "Presentation quality (rendered from this 3D)",
      "Could you render this 3D model to produce a client-presentable image?",
      "1 — Unusable for presentation rendering", "7 — Ready for high-quality client rendering"
    );
    scale7(form,
      "Overall 3D quality",
      "Holistic score for the 3D output.",
      "1 — Very poor", "7 — Excellent"
    );

    radio(form,
      "Would you use this 3D output in a real project?",
      [
        "Yes — I would use it as-is",
        "Yes — after some cleanup (less than 2 hours)",
        "Yes — after significant cleanup (more than 2 hours)",
        "No — it would be faster to model from scratch"
      ]
    );

    radio(form,
      "Which export format would be most useful for your workflow?",
      ["GLB / glTF", "OBJ + MTL", "FBX", "IFC (BIM)", "USDZ (AR)", "3DM (Rhino)", "Other"]
    );

    shortText(form, "If 'Other' export format — please specify", "", false);

    radio(form,
      "How much cleanup work would be required before this 3D model is usable in your workflow?",
      [
        "None — usable immediately",
        "Minor (under 30 minutes of fixing)",
        "Moderate (1–4 hours of fixing)",
        "Major (more than 4 hours of fixing)",
        "Complete redo — it would be faster to rebuild from scratch"
      ]
    );

    // Open text
    sectionHeader(form, "3D open-ended feedback");

    longText(form,
      "Describe any mesh or geometry problems you observed",
      "Holes, wrong topology, incorrect object boundaries, missing surfaces, etc. Leave blank if none."
    );

    longText(form,
      "Describe any texture or material problems you observed",
      "Blurry areas, colour shifts, missing textures, tiling artefacts, etc. Leave blank if none."
    );

    longText(form,
      "What elements are missing from the 3D model that should be present?",
      "E.g. specific furniture pieces, doors, windows, ceiling fixtures, lighting elements."
    );

    longText(form,
      "What unexpected or incorrect elements appeared in the 3D model?",
      "Objects that were not in the original room or the 2D variant."
    );

    longText(form,
      "Any additional observations about the 3D output?",
      "Technical, aesthetic, or practical. What would make you choose this over alternative reconstruction methods?"
    );
  }


  // ── Done ──────────────────────────────────────────────────────────────────

  var url = form.getPublishedUrl();
  var editUrl = form.getEditUrl();

  Logger.log("==============================================");
  Logger.log("Form created successfully.");
  Logger.log("Share with participants: " + url);
  Logger.log("Edit the form:           " + editUrl);
  Logger.log("==============================================");

  // Also write URLs to a Google Doc for easy reference.
  var doc = DocumentApp.create(STUDY_TITLE + " — Form URLs");
  var body = doc.getBody();
  body.appendParagraph("Participant URL (share this):").setBold(true);
  body.appendParagraph(url);
  body.appendParagraph("");
  body.appendParagraph("Edit URL (researcher only):").setBold(true);
  body.appendParagraph(editUrl);
  body.appendParagraph("");
  body.appendParagraph("Viewer URL configured in the form:").setBold(true);
  body.appendParagraph(VIEWER_BASE_URL);
  doc.saveAndClose();

  Logger.log("URLs also saved to Google Doc: " + doc.getUrl());
}
