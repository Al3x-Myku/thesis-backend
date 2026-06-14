# thesis-backend

Deci, basicamente ce face asta: dai o poza cu o camera + un mood board si sistemul iti genereaza variante 2D ale camerei cu paleta de culori si stilul mobilierului schimbate separat, dupa care poti sa "commitui" varianta preferata la un model 3D .glb. E proiectul de teza, nu judecati codul prea tare.

---

## Cum functioneaza (pe scurt)

```
Poza camera → scenes router → scene_service → salveaza input.png → task Celery

Celery worker → reconstructor_pipeline.py:full_reconstruction()
   1. detect_objects()  → D-FINE (sau fallback HuggingFace) → crops + coordonate
   2. build_mesh()      → BiRefNet scoate background → Hunyuan3D-2 face shape → Hunyuan3D-2 face textura
   3. position_meshes() → DepthPro estimeaza adancime → asambleaza scena → .glb final
```

### Design Zone (ramura design-zone)

```
POST /scenes/{id}/design/moodboard → task analyze_moodboard
   moodboard.py: extrage paleta CIELAB (KMeans) + stilul mobilierului (D-FINE + CLIP)
   → DesignSpec{ Palette, FurnitureSpec }

POST .../variants?n=4&mode=full → task generate_2d_variants
   restyle_2d.py: conditionare structura (depth + Canny + bounding boxes)
   + SD1.5 + ControlNet(depth) + IP-Adapter (imagine cu swatchuri de culori)
   + apply_palette_projection (transport Reinhard in spatiu LAB)
   → N variante PNG

mode=full          → IP-Adapter primeste paleta + cropuri mobilier din moodboard (style bleed)
mode=palette_only  → IP-Adapter primeste doar paleta → mobilierul din camera original ramas intact

POST .../variants/{vid}/feedback   → marcheaza elemente liked/disliked (dict blocat in DB)
POST .../variants/{vid}/regenerate → regenereaza pastrnd sloturile blocate pixel-exact (M4)
POST .../variants/{vid}/commit     → task commit_3d → commit_reconstruction() → .glb (M5)
```

**Disentanglement**: paleta CIELAB = canalul de aparenta (IP-Adapter + `apply_palette_projection`). Mobilierul = canalul de structura (conditionare ControlNet). Sunt injectate independent si pot fi editate independent. La commit 3D, paleta re-textureza pass-ul de paint Hunyuan3D-2 fara sa regenereze geometria.

---

## Structura repo

```
app/
  reconstructor_pipeline.py   # pipeline GPU complet; dict _models_cache; env var MESH_ENGINE
  dfine_wrapper.py             # wrapper subprocess D-FINE (fallback HF Transformers inclus)
  instantmesh_wrapper.py       # brat ablatie InstantMesh (subprocess, ca dfine_wrapper)
  panoramic_depth.py           # MTPano / DepthPro pentru adancime panorame 360
  moodboard.py                 # dataclass-uri DesignSpec, extract_palette, extract_furniture_style
  restyle_2d.py                # apply_palette_projection, build_structure_conditioning, generate_variants
  scene_geometry.py            # primitive geometrie (intrinsics, RANSAC podea, scalare metrica)
  wall_pipeline.py             # path pentru panorame 360, full_reconstruction_panoramic()
  tasks.py                     # definitii taskuri Celery
  routers/                     # routere FastAPI: scenes, auth, users, debug, design, evaluation
  services/                    # layer de orchestrare
  repositories/                # acces DB cu SQLModel
  models/                      # tabele SQLModel (Scene, User, DesignSession, Variant, etc.)
D-FINE/                        # detector obiecte vendored (Objects365, rulat ca subprocess)
Hunyuan3D-2/                   # generator 3D vendored
InstantMesh/                   # brat ablatie — cloneaza TencentARC/InstantMesh aici
scripts/                       # harness evaluare benchmark + script batch pipeline
tests/                         # suite de teste
docs/PLAN.md                   # metodologie si breakdown milestone-uri
```

---

## Setup mediu

Testat pe RTX 5080 16 GB, CUDA 12.8, Ubuntu 24.04. Pe alte configuratii nu garantez nimic.

```bash
# Creeaza mediu conda cu stiva ML completa
conda create -n ml python=3.12
conda activate ml

# PyTorch (CUDA 12.8 / Blackwell)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Dependente backend
pip install -r requirements_backend.txt

# Stiva ML completa
pip install -r requirements_all.txt

# Dependente runtime suplimentare
pip install trimesh==4.6.12 calflops

# Hunyuan3D ops CUDA custom (compileaza rasterizer + renderer)
pip install -e Hunyuan3D-2

# InstantMesh (brat ablatie, optional)
git clone https://github.com/TencentARC/InstantMesh
pip install -r InstantMesh/requirements.txt
```

### Variabile de mediu (`.env`)

```
SECRET_KEY=<string-random>
DATABASE_URL=mysql+pymysql://user:pass@host/dbname

# Optionale — valorile default sunt astea
DATA_DIR=./data
PIPELINE_DEVICE=cuda:0
DFINE_ROOT=./D-FINE
DFINE_CONFIG=configs/dfine/objects365/dfine_hgnetv2_x_obj365.yml
DFINE_CHECKPT=weights/dfine_x_obj365.pth
HUNYUAN_SHAPEDIR=tencent/Hunyuan3D-2
HUNYUAN_PAINTDIR=tencent/Hunyuan3D-2

# Ablatie si panorame
MESH_ENGINE=hunyuan                    # sau instantmesh
INSTANTMESH_ROOT=./InstantMesh
MTPANO_MODEL_ID=Evergreen0929/MTPano
```

**Greutati D-FINE**: descarca `dfine_x_obj365.pth` din [D-FINE releases](https://github.com/Peterande/D-FINE/releases) si pune-l la `D-FINE/weights/dfine_x_obj365.pth`. Daca lipseste sau subprocesul pica, `detect_objects` foloseste automat fallback-ul `ustc-community/dfine_x_obj365` via HuggingFace Transformers.

---

## Rulare server

```bash
# API (dev, auto-reload)
uvicorn app.main:app --reload

# Worker Celery (necesar pentru orice task de reconstructie sau design)
celery -A app.celery_app.celery_app worker --loglevel=info
```

Broker si result backend sunt ambele `DATABASE_URL` (SQLAlchemy). Nu trebuie Redis sau RabbitMQ.

---

## Rulare batch (mai multe perechi poza+moodboard)

```bash
python scripts/batch_run.py \
  --pairs-dir pairs/ \
  --api http://localhost:8000 \
  --email email@tau.com \
  --password parola \
  --variants 4 \
  --mode full \
  --commit-variant 0
```

Structura folder perechi:
```
pairs/
  001/
    room.jpg         ← poza camerei
    moodboard/
      board_1.jpg    ← una sau mai multe imagini moodboard
      board_2.jpg
  002/
    room.jpg
    moodboard/
      board_1.jpg
```

Flaguri utile:
- `--mode palette_only` — doar paleta de culori, fara stilul mobilierului din moodboard
- `--commit-variant -1` — sari peste commit 3D (mult mai rapid)
- `--engine instantmesh` — foloseste InstantMesh in loc de Hunyuan3D-2 pentru 3D
- `--dry-run` — vezi ce s-ar procesa fara sa faci apeluri API

Scriptul e **resumabil** — salveaza `out/meta.json` dupa fiecare pas. Daca il opresti la jumatate, la urmatoarea rulare sare peste pasii deja facuti.

---

## Rulare teste

### Teste CPU (fara GPU)

Acopera logica pura, ruleaza pe orice masina cu numpy/Pillow/scikit-learn:

```bash
pytest tests/test_scene_geometry.py tests/test_moodboard.py tests/test_restyle_2d.py -v

# Necesita doar:
pip install numpy pillow scikit-learn scipy trimesh pytest
```

### Teste GPU (necesita mediu ML complet)

Valideaza pipeline-urile live — embeddings CLIP, restyle difuziv (SD1.5 + ControlNet + IP-Adapter), detectie D-FINE, estimare adancime:

```bash
conda activate ml
pytest tests/test_design_gpu.py -v
```

| Clasa | Teste | Ce ruleaza pe GPU |
|---|---|---|
| `TestM2MoodboardAnalysis` | 4 | `extract_palette`, `extract_furniture_style` (D-FINE + CLIP), roundtrip `analyze_moodboard` |
| `TestM3RestyLe2D` | 3 | `build_structure_conditioning` (depth + detectie), `generate_variants` (difuzie), assert schimbare culoare |
| `TestM4FeedbackFreeze` | 2 | Identitate pixeli slot blocat, divergenta varianta deblocata |
| `TestM5PaletteUtils` | 5 | `palette_to_shell_colors`, `position_meshes(shell_colors=, out_subdir=)`, compat backwards |
| `TestDesignFlowIntegration` | 1 | Flux complet moodboard → spec → 3 variante end-to-end |

Suite completa (CPU + GPU):

```bash
conda activate ml
pytest tests/ -q
# Expected: 43 passed
```

---

## Brat ablatie InstantMesh

InstantMesh (TencentARC/InstantMesh, 2024) e un al doilea path de generare 3D care produce **harti de textura PBR UV** (Albedo / Normal / AO) in loc de culorile pe vertex de la Hunyuan3D-2. Conteaza pentru evaluarea studentilor de arhitectura — BIM readiness si re-texturabilitate downstream sunt mult mai mari cu output UV-mapped.

### Comutare intre motoare

```bash
# Foloseste InstantMesh pentru toata generarea de mesh-uri
MESH_ENGINE=instantmesh uvicorn app.main:app --reload

# Foloseste Hunyuan3D-2 (default)
MESH_ENGINE=hunyuan uvicorn app.main:app --reload
```

| | Hunyuan3D-2 | InstantMesh |
|---|---|---|
| Geometrie | Mesh watertight (marching cubes) | Mesh reconstruit LRM |
| Textura | Culori pe vertex (paint pass) | Harti PBR UV (Albedo / Normal / AO) |
| BIM readiness | Scazut | Mai ridicat (UV-mapped, importabil in Revit/Blender) |
| Viteza | ~60s | ~30s |

### Context SOTA 2026

- **Mesh-Pro** (CVPR 2026, arXiv:2603.00526, Tencent): framework RL asincron pentru generare mesh (ARPO); de 3.75× mai rapid decat metodele anterioare. Succesor InstantMesh.
- **Hunyuan3D-2.1** (2026): adauga backbone dataset "LATTICE", detalii suprafata mai bune, varianta PBR. Cale directa de upgrade de la greutatile `tencent/Hunyuan3D-2` actuale.
- **FreeMesh** (ICML 2025, arXiv:2505.13573): compresie coordonate plug-in pentru MeshAnything V2; imbunatateste compactitatea mesh-urilor.

---

## Reconstructie la scara camerei (panorame)

Path-ul `wall_pipeline.py` trateaza panorame 360° equirectangulare. Descompune panorama in vederi perspective per-perete, ruleaza detectie + Hunyuan3D-2 (sau InstantMesh) pe fiecare crop de mobilier detectat, apoi asambleaza un shell de camera (pereti + podea + tavan) cu obiectele plasate.

### Estimare adancime: MTPano

`app/panoramic_depth.py` furnizeaza adancime metrica pentru panorame folosind **MTPano** (SIGGRAPH 2026, arXiv:2602.05330). MTPano e un model fondational multi-task antrenat specific pe imagini equirectangulare — trateaza distorsiunea proiectiei sferice care face modelele standard de adancime perspective inaccurate langa marginile si polii panoramei.

Lantul de prioritate in `full_reconstruction_panoramic()`:
```
1. MTPano (aware de panorame, metric)         ← sursa primara noua
2. Fisier GT depth (pano_folder/full/depth.png) ← doar pentru benchmark Structured3D
3. Fara adancime (distante perete folosite)   ← fallback existent
```

### Context SOTA 2026 (panorame)

- **PanoVGGT** (CVPR 2026, arXiv:2603.17571): ia mai multe panorame equirectangulare → nori de puncte 3D consistent globali + adancime + pose camera intr-un singur forward pass. Cel mai bun pentru capturi multi-vedere.
- **MTPano** (SIGGRAPH 2026, arXiv:2602.05330): o singura panorama → adancime + normale suprafata + semantica. **Sursa primara actuala** pentru capturi cu o singura panorama.
- **HY-World 2.0** (arXiv:2604.14268, Tencent): generare lume completa dintr-o panorama → mesh + Gaussiene 3D. Exporta in Unity/Unreal.

---

## Sistem validare umana

Instrument de cercetare structurat pentru colectarea evaluarilor de la studenti de arhitectura. Proiectat pentru utilizare paralela cu Google Forms / Qualtrics.

### Modelul de acces

```
Cercetator (JWT) → POST /evaluation/sessions
                 ← { token, participant_url_hint: "/evaluation/{token}/view" }

Distribuie URL participantilor (nu necesita cont)
   GET /evaluation/{token}/view → pagina HTML cu toate variantele + viewer 3D

Exporta:
   GET /evaluation/export/profiles.csv       → demografice
   GET /evaluation/export/variants_2d.csv    → evaluari per-varianta
   GET /evaluation/export/variant_sets.csv   → comparatie cross-varianta
   GET /evaluation/export/commit_3d.csv      → evaluari model 3D
   GET /evaluation/export/all.csv            → toate tabelele concatenate
```

### Cele 4 formulare de evaluare

#### Formular 1 — Profil evaluator (demografice, completat o data)

Campuri: interval varsta, rol profesional, ani experienta, nivel educatie, specializare, tara educatie, software folosit, 3 scale Likert 1-5 (familiaritate cu instrumente AI, modelare 3D, software design interior).

#### Formular 2 — Evaluare varianta 2D (completat o data per varianta)

18 itemi Likert 1-7 pe 4 canale:

| Canal | Itemi |
|---|---|
| Aparenta / paleta | Calitate estetica, armonie culori, fidelitate paleta fata de moodboard, atmosfera/dispozitie, adecvare temperatura culoare |
| Structura / mobilier | Pastrare layout spatial, consistenta stil mobilier, plauzabilitate plasament mobilier, acuratete scala si proportii |
| Realism | Fotorealism, plauzabilitate iluminat, calitate materiale/suprafete, calitate umbre si reflectii |
| Profesional | Adecvare profesionala, prezentabilitate client, inovatie si creativitate, coerenta design |

Plus itemi categorici si 4 campuri text deschis: aspectul cel mai atragator, aspectul cel mai problematic, sugestii de design, elemente care dezvaluie originea AI.

#### Formular 3 — Comparatie set variante (completat o data dupa toate variantele)

Proba esentiala de cercetare pentru afirmatia de disentanglement:
- **Ranking**: lista de preferinta ordonata a ID-urilor de variante
- **Perceptie disentanglement**: `palette_change_perceived`, `furniture_change_perceived`, `perceived_what_changed` (dict per-canal), `disentanglement_clarity` (1-7), `palette_axis_control_confidence` (1-7), `furniture_axis_control_confidence` (1-7)
- **Proba scurgere**: `cross_channel_leakage_observed` (bool) + descriere — masoara daca o schimbare intentionata pentru un canal a alterat neintenționat celalalt
- Evaluare instrument AI: utilitate, incredere, usurinta integrare workflow, timp economisit vs manual

#### Formular 4 — Evaluare commit 3D (completat dupa viewer-ul 3D)

18 itemi: acuratete geometrie (completitudine mesh, acuratete dimensiuni spatiale, acuratete geometrie mobilier, prezenta artefacte), aparenta (calitate textura, reprezentare materiale, fidelitate paleta 3D, nivel detaliu suprafata), fidelitate 2D→3D (4 itemi), aplicabilitate profesionala (utilizabilitate pentru modelare ulterioara, BIM readiness, calitate prezentare, scor global).

### Variabile cheie pentru analiza statistica

| Variabila | Tabel | Utilizare |
|---|---|---|
| `palette_fidelity_to_moodboard` (1-7) | variant_2d | Fidelitate canal paleta |
| `spatial_layout_preservation` (1-7) | variant_2d | Fidelitate canal structura |
| `disentanglement_clarity` (1-7) | variant_sets | Independenta perceputa a axelor |
| `cross_channel_leakage_observed` (bool) | variant_sets | Scurgere masurata |
| `palette_change_perceived` / `furniture_change_perceived` | variant_sets | Proba de perceptie |
| `engine_type` (`diffusion` / `baseline`) | variant_2d | Eticheta brat ablatie |
| `presentation_position` (1..N) | variant_2d | Covariabila bias pozitie |
| `fidelity_2d_to_3d` (1-7) | commit_3d | Fidelitate pipeline 2D→3D |

---

## Note tehnice importante

- **Disciplina memorie GPU**: modele mutate pe/de pe device in jurul fiecarui stage (`model.to(device)` inainte, `model.to("cpu")` + `cleanup_gpu()` dupa). Pastreaza asta cand adaugi stage-uri.
- **Import-uri lazy**: `app/moodboard.py` si `app/restyle_2d.py` amana toate import-urile grele (torch, diffusers, CLIP, `reconstructor_pipeline`) pana la momentul apelului, asa ca tier-ul web ramane importabil pe o masina fara GPU.
- **Lant fallback difuzie**: SD1.5 + ControlNet + IP-Adapter → retry fara IP-Adapter → `apply_palette_projection` pur (baseline CPU). Path-ul CPU e si bratul de ablatie histogram-matching pentru evaluare.
- **ID-uri clase obiecte**: pipeline-ul foloseste ID-uri **Objects365** peste tot (D-FINE e checkpoint-ul obj365). Vezi `OBJ365_NAMES` in `reconstructor_pipeline.py`.
- **Artefacte scene**: `data/user_<id>/scene_<id>/` contine `input.png`, `crops/`, `meshes/`, `structure/`, `variants/`, `committed/`, `final/scene_positioned.glb`. Nu restructura asta fara sa actualizezi endpointurile de download.
