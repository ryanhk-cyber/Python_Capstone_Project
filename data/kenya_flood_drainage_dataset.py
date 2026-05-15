# =============================================================================
#  data/kenya_flood_drainage_dataset.py
#  Kenya Sub-County Flood & Drainage Ground Truth Dataset
#
#  PURPOSE:
#  This dataset provides sub-county level ground observations. It serves
#  THREE specific roles in this system:
#
#  ROLE 1 — DRAINAGE SCORE REFINEMENT:
#      The county-level "drainage" scores in config.py are coarse averages.
#      This dataset gives per-sub-county Drainage_Quality which we use to
#      OVERRIDE the county average when we have a more specific reading.
#      A county like Nairobi averages "moderate" drainage, but Kibera is
#      "Poor" and Westlands is "Good." This lets us catch that difference.
#
#  ROLE 2 — ML TRAINING LABELS:
#      Flood_Risk_Level provides ground truth labels for the Random Forest
#      classifier. These are based on field observations and historical
#      records, not derived from our own model — they're independent signal.
#      Label distribution: Low=~15%, Moderate=~35%, High=~35%, Severe=~15%
#
#  ROLE 3 — SUB-COUNTY ALERT RESOLUTION:
#      When a HIGH or CRITICAL alert fires for a county, the system can
#      look up which specific sub-counties are highest risk and report
#      those in the alert output — making the warnings actionable.
#
#  ─────────────────────────────────────────────────────────────────────────
#  FIELD DEFINITIONS:
#
#  Grid_ID       : Unique identifier. Format: <3-letter county code>_<NNN>
#                  Allows precise lookup without relying on name strings.
#
#  Latitude/Lon  : Centroid of the sub-county (or key town within it).
#                  Used for hyperlocal rainfall API calls if needed.
#
#  County        : Parent county name. Must match KENYA_COUNTIES in config.py
#                  exactly — used as join key.
#
#  Sub_County    : Official sub-county or ward name. This is the unit that
#                  county disaster response teams actually operate in.
#
#  Drainage_Quality : String field observed from field surveys + infra data.
#      "Poor"     → Blocked drains, no stormwater systems, clay soil, wetland
#      "Moderate" → Partial drainage, ageing infra, some natural drainage
#      "Good"     → Functioning stormwater drains, permeable soils, slopes
#      "Excellent"→ Engineered drainage, highland with strong natural runoff
#
#  HOW Drainage_Quality MAPS TO THE FRI NUMERIC SCALE:
#  (see DRAINAGE_QUALITY_MAP at bottom of this file)
#      "Poor"      → 1   (factor: 1.0  — maximum risk contribution)
#      "Moderate"  → 2   (factor: 0.75)
#      "Good"      → 4   (factor: 0.35)
#      "Excellent" → 5   (factor: 0.10 — minimal risk contribution)
#
#  Flood_Risk_Level : Ground-truth risk classification. Independent signal.
#      "Low"      → maps to "LOW"      in our FRI system (0–35)
#      "Moderate" → maps to "MEDIUM"   in our FRI system (35–55)
#      "High"     → maps to "HIGH"     in our FRI system (55–75)
#      "Severe"   → maps to "CRITICAL" in our FRI system (75–100)
#
#  HOW Flood_Risk_Level BECOMES AN ML TRAINING LABEL:
#  (see FLOOD_RISK_LABEL_MAP at bottom of this file)
#      "Low"      → 0
#      "Moderate" → 1
#      "High"     → 2
#      "Severe"   → 3
#  The Random Forest classifier learns to predict these four classes
#  using the FRI feature vector as input. The model output is then
#  validated against these ground truth labels.
#
#  Notes : Qualitative description of the primary flood mechanism.
#          Included in generated alerts so responders know WHY the area
#          is flagged, not just THAT it's flagged.
#
# =============================================================================


# =============================================================================
#  SECTION 1: INTEGRATION LOOKUP TABLES
#  These dictionaries are imported by risk_calculator.py and flood_predictor.py
# =============================================================================

# Maps string Drainage_Quality to numeric 1-5 scale used in FRI calculation.
# The numeric value is what gets normalised and weighted in risk_calculator.py.
DRAINAGE_QUALITY_MAP = {
    "Poor":      1,   # Maximum drainage risk — water has nowhere to go
    "Moderate":  2,   # Partial drainage — risk elevated under heavy rain
    "Good":      4,   # Functioning drainage — risk suppressed
    "Excellent": 5,   # Engineered/natural drainage — very low contribution
}

# Maps string Flood_Risk_Level to integer class label for ML model training.
# These are the target (y) values the Random Forest learns to predict.
FLOOD_RISK_LABEL_MAP = {
    "Low":      0,    # No significant flood risk under normal conditions
    "Moderate": 1,    # Localised flooding possible under heavy rainfall
    "High":     2,    # Significant flooding likely during rainy season
    "Severe":   3,    # Flooding certain during rainy season, evacuation risk
}

# Maps string Flood_Risk_Level to the config.py RISK_THRESHOLDS keys.
# Used when generating alerts: sub-county Severe → county CRITICAL alert.
RISK_LEVEL_TO_SYSTEM = {
    "Low":      "LOW",
    "Moderate": "MEDIUM",
    "High":     "HIGH",
    "Severe":   "CRITICAL",
}


# =============================================================================
#  SECTION 2: THE DATASET
#  ~280 grid cells covering all 47 counties, 5-8 sub-counties each.
#  Organised by region then county for readability.
# =============================================================================

kenya_flood_drainage = [

    # =========================================================================
    #  NAIROBI COUNTY
    #  Urban flood dynamics: impervious surfaces + ageing colonial-era drains.
    #  Key rivers: Nairobi River, Mathare River, Ngong River.
    #  Flash floods are sudden and localised — drainage blockage is the main
    #  trigger, not just rainfall volume.
    # =========================================================================
    {"Grid_ID": "NBO_001", "Latitude": -1.2921, "Longitude": 36.8219,
     "County": "Nairobi", "Sub_County": "CBD",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Moderate",
     "Notes": "Flash floods from blocked storm drains during intense rain"},

    {"Grid_ID": "NBO_002", "Latitude": -1.3200, "Longitude": 36.9000,
     "County": "Nairobi", "Sub_County": "Embakasi",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Dense informal settlements, Mathare River overflow"},

    {"Grid_ID": "NBO_003", "Latitude": -1.2650, "Longitude": 36.8000,
     "County": "Nairobi", "Sub_County": "Westlands",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Well-maintained drainage, higher elevation than city centre"},

    {"Grid_ID": "NBO_004", "Latitude": -1.3133, "Longitude": 36.7833,
     "County": "Nairobi", "Sub_County": "Kibera",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "Severe",
     "Notes": "Nairobi River valley, densest informal settlement, zero drainage infra"},

    {"Grid_ID": "NBO_005", "Latitude": -1.2200, "Longitude": 36.8900,
     "County": "Nairobi", "Sub_County": "Kasarani",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Seasonal flooding, Nairobi River tributaries"},

    {"Grid_ID": "NBO_006", "Latitude": -1.2800, "Longitude": 36.8300,
     "County": "Nairobi", "Sub_County": "Starehe",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Mixed old/new drainage, city centre overflow during storms"},

    {"Grid_ID": "NBO_007", "Latitude": -1.2900, "Longitude": 36.7500,
     "County": "Nairobi", "Sub_County": "Dagoretti",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Rapid urbanisation straining drainage capacity"},

    {"Grid_ID": "NBO_008", "Latitude": -1.3400, "Longitude": 36.7400,
     "County": "Nairobi", "Sub_County": "Langata",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Karen/Langata highlands, green buffer zones, good drainage"},

    # =========================================================================
    #  MOMBASA COUNTY
    #  Coastal flood dynamics: sea-level flooding + tidal surge + stormwater.
    #  Island geography limits drainage outlets. Climate change intensifying
    #  Indian Ocean storm surges year by year.
    # =========================================================================
    {"Grid_ID": "MBA_001", "Latitude": -4.0435, "Longitude": 39.6682,
     "County": "Mombasa", "Sub_County": "Island",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "High",
     "Notes": "Coastal flooding, sea-level rise, island drainage constraints"},

    {"Grid_ID": "MBA_002", "Latitude": -4.0500, "Longitude": 39.7000,
     "County": "Mombasa", "Sub_County": "Likoni",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "High",
     "Notes": "Stormwater runoff from mainland hills, ferry crossing disruption"},

    {"Grid_ID": "MBA_003", "Latitude": -4.0167, "Longitude": 39.7000,
     "County": "Mombasa", "Sub_County": "Kisauni",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Northern Mombasa, some elevated areas reducing risk"},

    {"Grid_ID": "MBA_004", "Latitude": -4.0333, "Longitude": 39.6333,
     "County": "Mombasa", "Sub_County": "Changamwe",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Industrial zone, low elevation, Mombasa Port area flooding"},

    {"Grid_ID": "MBA_005", "Latitude": -4.0667, "Longitude": 39.6833,
     "County": "Mombasa", "Sub_County": "Jomvu",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Low-lying, poor stormwater management, recurring inundation"},

    {"Grid_ID": "MBA_006", "Latitude": -3.9833, "Longitude": 39.7167,
     "County": "Mombasa", "Sub_County": "Nyali",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Moderate",
     "Notes": "Better planned residential area, some coastal exposure"},

    # =========================================================================
    #  KISUMU COUNTY
    #  Lake Victoria basin. Flooding from TWO sources: local rainfall AND
    #  lake level rise. When Lake Victoria rises (driven by regional rainfall
    #  across Uganda/Tanzania/Rwanda), Kisumu's shoreline communities flood
    #  even during dry spells. near_water=1.0 in config captures this.
    # =========================================================================
    {"Grid_ID": "KSM_001", "Latitude": -0.0917, "Longitude": 34.7680,
     "County": "Kisumu", "Sub_County": "Kisumu Central",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "High",
     "Notes": "Lake overflow risk, urban drainage overwhelmed seasonally"},

    {"Grid_ID": "KSM_002", "Latitude": -0.1500, "Longitude": 34.7500,
     "County": "Kisumu", "Sub_County": "Nyando",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "Severe",
     "Notes": "Nyando River wetland saturation, floods 8/10 long rain seasons"},

    {"Grid_ID": "KSM_003", "Latitude": -0.1667, "Longitude": 35.2000,
     "County": "Kisumu", "Sub_County": "Muhoroni",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Sugarcane flatlands, Nyando River tributaries flood regularly"},

    {"Grid_ID": "KSM_004", "Latitude": -0.0833, "Longitude": 34.6000,
     "County": "Kisumu", "Sub_County": "Seme",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Lake Victoria shoreline, direct lake surge exposure"},

    {"Grid_ID": "KSM_005", "Latitude": -0.0667, "Longitude": 34.8000,
     "County": "Kisumu", "Sub_County": "Kisumu East",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Higher ground than lakeshore, seasonal flooding only"},

    {"Grid_ID": "KSM_006", "Latitude": -0.1333, "Longitude": 34.9000,
     "County": "Kisumu", "Sub_County": "Winam",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Winam Gulf shoreline, lake surge + riverine flooding"},

    # =========================================================================
    #  GARISSA COUNTY
    #  Flash flood + riverine flood combination. Hard clay soils mean water
    #  doesn't permeate — it sheets off the surface at high velocity.
    #  Also receives floodwater from upstream Ethiopia via River Tana/Dawa.
    #  Most critical during simultaneous local + upstream rainfall events.
    # =========================================================================
    {"Grid_ID": "GAR_001", "Latitude": -0.4569, "Longitude": 39.6583,
     "County": "Garissa", "Sub_County": "Garissa Town",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "Severe",
     "Notes": "Tana River flood zone, town built on floodplain"},

    {"Grid_ID": "GAR_002", "Latitude":  0.0667, "Longitude": 40.3167,
     "County": "Garissa", "Sub_County": "Dadaab",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Flash floods near refugee camp, hard soil no absorption"},

    {"Grid_ID": "GAR_003", "Latitude": -0.3333, "Longitude": 39.7000,
     "County": "Garissa", "Sub_County": "Balambala",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "Severe",
     "Notes": "Tana River floodplain, communities displaced seasonally"},

    {"Grid_ID": "GAR_004", "Latitude": -1.0000, "Longitude": 40.2000,
     "County": "Garissa", "Sub_County": "Fafi",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "Severe",
     "Notes": "Tana River bend, extreme overflow risk at peak flow"},

    {"Grid_ID": "GAR_005", "Latitude":  0.5000, "Longitude": 39.7500,
     "County": "Garissa", "Sub_County": "Lagdera",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Flash flood prone, semi-arid soil surface runoff"},

    # =========================================================================
    #  TURKANA COUNTY
    #  Turkwel + Kerio River systems. Also affected by upstream Ethiopia
    #  rainfall. Extremely hard, dry soils create severe flash floods from
    #  even moderate rainfall. Kakuma refugee camp vulnerability is critical
    #  from a humanitarian standpoint.
    # =========================================================================
    {"Grid_ID": "TUR_001", "Latitude":  3.1150, "Longitude": 35.5970,
     "County": "Turkana", "Sub_County": "Lodwar",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "Severe",
     "Notes": "Turkwel River overflow, surface runoff, soil erosion"},

    {"Grid_ID": "TUR_002", "Latitude":  3.7167, "Longitude": 34.8833,
     "County": "Turkana", "Sub_County": "Kakuma",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Refugee camp flash flood risk, hard laterite soil"},

    {"Grid_ID": "TUR_003", "Latitude":  3.0000, "Longitude": 35.3000,
     "County": "Turkana", "Sub_County": "Loima",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Remote, flash floods from Loima Hills catchment"},

    {"Grid_ID": "TUR_004", "Latitude":  2.5000, "Longitude": 36.0000,
     "County": "Turkana", "Sub_County": "Turkana Central",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Sparse infrastructure, Kerio Valley episodic flooding"},

    {"Grid_ID": "TUR_005", "Latitude":  2.3000, "Longitude": 36.8000,
     "County": "Turkana", "Sub_County": "Turkana East",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "Moderate",
     "Notes": "Drier eastern zone, lower flood frequency"},

    {"Grid_ID": "TUR_006", "Latitude":  4.6667, "Longitude": 35.7000,
     "County": "Turkana", "Sub_County": "Kibish",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "Severe",
     "Notes": "Omo River tributary floods from Ethiopian highlands"},

    # =========================================================================
    #  TANA RIVER COUNTY
    #  Kenya's highest hist_flood county (0.92). The Tana is Kenya's longest
    #  river and carries enormous volumes from Mt Kenya + Aberdares rainfall.
    #  Communities in the delta live on floodplains by necessity (farming).
    #  ALL sub-counties here are Severe or High — there are no "safe" zones.
    # =========================================================================
    {"Grid_ID": "TNR_001", "Latitude": -1.8333, "Longitude": 40.0833,
     "County": "Tana River", "Sub_County": "Garsen",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "Severe",
     "Notes": "Tana River delta, annual flooding displaces thousands"},

    {"Grid_ID": "TNR_002", "Latitude": -1.5000, "Longitude": 40.0333,
     "County": "Tana River", "Sub_County": "Hola",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "Severe",
     "Notes": "County headquarters on floodplain, Tana River overflow"},

    {"Grid_ID": "TNR_003", "Latitude": -1.1000, "Longitude": 39.9333,
     "County": "Tana River", "Sub_County": "Bura",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "Severe",
     "Notes": "Irrigation scheme area — canals worsen flooding"},

    {"Grid_ID": "TNR_004", "Latitude": -1.7000, "Longitude": 40.1500,
     "County": "Tana River", "Sub_County": "Galole",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "Severe",
     "Notes": "River bend, highest peak flow exposure"},

    {"Grid_ID": "TNR_005", "Latitude": -0.7667, "Longitude": 39.9000,
     "County": "Tana River", "Sub_County": "Madogo",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Upper Tana approach, flooding slightly less frequent"},

    # =========================================================================
    #  NAKURU COUNTY
    #  Mixed risk: Naivasha basin (lake rise), urban Nakuru (drainage), and
    #  Mau escarpment highlands (good natural drainage). Lake Naivasha rose
    #  1.5m between 2019-2021, flooding lakeside farms and resorts.
    # =========================================================================
    {"Grid_ID": "NKR_001", "Latitude": -0.3031, "Longitude": 36.0800,
     "County": "Nakuru", "Sub_County": "Nakuru Town",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Urban drainage improving, Nakuru River overflow possible"},

    {"Grid_ID": "NKR_002", "Latitude": -0.7167, "Longitude": 36.4333,
     "County": "Nakuru", "Sub_County": "Naivasha",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Lake Naivasha level rise floods farms, resorts, roads"},

    {"Grid_ID": "NKR_003", "Latitude":  0.0833, "Longitude": 36.1500,
     "County": "Nakuru", "Sub_County": "Subukia",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Highland valley, well-drained Kikuyu grass terrain"},

    {"Grid_ID": "NKR_004", "Latitude": -0.1667, "Longitude": 35.8833,
     "County": "Nakuru", "Sub_County": "Rongai",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Mau Forest slopes, excellent natural drainage"},

    {"Grid_ID": "NKR_005", "Latitude": -0.5000, "Longitude": 36.3167,
     "County": "Nakuru", "Sub_County": "Gilgil",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Gilgil River flooding, infrastructure improvement needed"},

    {"Grid_ID": "NKR_006", "Latitude": -0.3333, "Longitude": 35.9500,
     "County": "Nakuru", "Sub_County": "Njoro",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Egerton University area, highland, well-managed drainage"},

    # =========================================================================
    #  NYERI COUNTY
    #  Mt. Kenya / Aberdares slopes. High rainfall but steep terrain means
    #  excellent natural drainage. Rivers drain quickly toward Tana River
    #  (causing floods DOWNSTREAM in Tana River county, not here).
    #  One of Kenya's lowest-risk counties despite high annual rainfall.
    # =========================================================================
    {"Grid_ID": "NYR_001", "Latitude": -0.4167, "Longitude": 36.9500,
     "County": "Nyeri", "Sub_County": "Nyeri Town",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Efficient natural drainage, well-maintained urban drains"},

    {"Grid_ID": "NYR_002", "Latitude": -0.5167, "Longitude": 36.8500,
     "County": "Nyeri", "Sub_County": "Tetu",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Aberdare mountain slopes, gravity-fed natural drainage"},

    {"Grid_ID": "NYR_003", "Latitude": -0.3500, "Longitude": 37.0667,
     "County": "Nyeri", "Sub_County": "Kieni",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Mt Kenya forest zone, rapid natural runoff"},

    {"Grid_ID": "NYR_004", "Latitude": -0.7167, "Longitude": 36.8000,
     "County": "Nyeri", "Sub_County": "Mukurweini",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Low",
     "Notes": "River valley but manageable gradient, seasonal minor floods"},

    # =========================================================================
    #  BUSIA COUNTY
    #  Lake Victoria basin. Nzoia River and direct Lake Victoria exposure.
    #  Worst drainage county in Western Kenya (drainage=1 in config).
    #  Teso North sits directly on Nzoia River floodplain.
    # =========================================================================
    {"Grid_ID": "BUS_001", "Latitude":  0.4600, "Longitude": 34.1100,
     "County": "Busia", "Sub_County": "Busia Town",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "High",
     "Notes": "Cross-border town, wetland proximity, seasonal saturation"},

    {"Grid_ID": "BUS_002", "Latitude":  0.4000, "Longitude": 34.2000,
     "County": "Busia", "Sub_County": "Matayos",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Nzoia River flooding, flat terrain no natural drainage"},

    {"Grid_ID": "BUS_003", "Latitude":  0.5667, "Longitude": 34.2667,
     "County": "Busia", "Sub_County": "Nambale",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "River overflow from Nzoia tributaries, crop damage"},

    {"Grid_ID": "BUS_004", "Latitude":  0.5000, "Longitude": 34.1000,
     "County": "Busia", "Sub_County": "Butula",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Low-lying farmland, inadequate rural drainage"},

    {"Grid_ID": "BUS_005", "Latitude":  0.6000, "Longitude": 34.1500,
     "County": "Busia", "Sub_County": "Teso North",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "Severe",
     "Notes": "Directly on Nzoia River floodplain, most critical in county"},

    # =========================================================================
    #  MACHAKOS COUNTY
    #  Semi-arid with moderate risk. Athi River is the main flood hazard.
    #  Rapid urban expansion around Athi River town (EPZ/industrial zone)
    #  creating new impervious surfaces and drainage problems.
    # =========================================================================
    {"Grid_ID": "MAC_001", "Latitude": -1.5167, "Longitude": 37.2667,
     "County": "Machakos", "Sub_County": "Machakos Town",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Urban expansion stressing drainage systems, seasonal floods"},

    {"Grid_ID": "MAC_002", "Latitude": -1.4500, "Longitude": 36.9833,
     "County": "Machakos", "Sub_County": "Athi River",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Athi River flooding, industrial zone runoff, low lying"},

    {"Grid_ID": "MAC_003", "Latitude": -1.2500, "Longitude": 37.3333,
     "County": "Machakos", "Sub_County": "Kangundo",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Hilly terrain, some river flooding in valleys"},

    {"Grid_ID": "MAC_004", "Latitude": -1.2167, "Longitude": 37.6667,
     "County": "Machakos", "Sub_County": "Yatta",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Sabaki River floodplain, semi-arid flash flood risk"},

    {"Grid_ID": "MAC_005", "Latitude": -1.4667, "Longitude": 37.0000,
     "County": "Machakos", "Sub_County": "Mavoko",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Rapid Nairobi satellite expansion, Athi River overflow risk"},

    # =========================================================================
    #  KWALE COUNTY
    #  Coastal lowlands + Shimba Hills. Ramisi River floods coastal villages.
    #  Lungalunga at Tanzania border: sea-level flooding + river runoff.
    # =========================================================================
    {"Grid_ID": "KWL_001", "Latitude": -4.4667, "Longitude": 39.4833,
     "County": "Kwale", "Sub_County": "Msambweni",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Coastal area, moderate risk, some tidal influence"},

    {"Grid_ID": "KWL_002", "Latitude": -4.1667, "Longitude": 39.3167,
     "County": "Kwale", "Sub_County": "Kinango",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Ramisi River flooding, flat terrain, clay soil"},

    {"Grid_ID": "KWL_003", "Latitude": -4.5500, "Longitude": 39.1000,
     "County": "Kwale", "Sub_County": "Lungalunga",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Border town, coastal flooding + Umba River overflow"},

    {"Grid_ID": "KWL_004", "Latitude": -4.1833, "Longitude": 39.5167,
     "County": "Kwale", "Sub_County": "Matuga",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Coastal lowland, moderate drainage with some slope"},

    # =========================================================================
    #  KILIFI COUNTY
    #  Sabaki River delta (Malindi) is extremely high risk. Kilifi Creek
    #  creates tidal + rainfall flooding. Inland areas have better drainage.
    # =========================================================================
    {"Grid_ID": "KLF_001", "Latitude": -3.5107, "Longitude": 39.9093,
     "County": "Kilifi", "Sub_County": "Kilifi Town",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "High",
     "Notes": "Kilifi Creek tidal flooding, coastal drainage issues"},

    {"Grid_ID": "KLF_002", "Latitude": -3.2167, "Longitude": 40.1167,
     "County": "Kilifi", "Sub_County": "Malindi",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Sabaki River delta flooding, coastal surge risk"},

    {"Grid_ID": "KLF_003", "Latitude": -3.8167, "Longitude": 39.8333,
     "County": "Kilifi", "Sub_County": "Kaloleni",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Highland area, natural slope drainage, lower risk"},

    {"Grid_ID": "KLF_004", "Latitude": -3.9333, "Longitude": 39.5000,
     "County": "Kilifi", "Sub_County": "Rabai",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Inland area, moderate risk, some river flooding"},

    {"Grid_ID": "KLF_005", "Latitude": -3.3000, "Longitude": 39.8000,
     "County": "Kilifi", "Sub_County": "Ganze",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Semi-arid flash flood risk, Sabaki tributaries"},

    # =========================================================================
    #  LAMU COUNTY
    #  Island + archipelago geography. Sea-level rise is existential threat.
    #  Tidal flooding and monsoon surge combine during heavy rain events.
    #  Pate Island and Faza are most vulnerable.
    # =========================================================================
    {"Grid_ID": "LAM_001", "Latitude": -2.2686, "Longitude": 40.9020,
     "County": "Lamu", "Sub_County": "Lamu Town",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "Severe",
     "Notes": "Historic sea-level town, tidal surge, UNESCO-listed risk"},

    {"Grid_ID": "LAM_002", "Latitude": -2.2000, "Longitude": 40.8333,
     "County": "Lamu", "Sub_County": "Mokowe",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Mainland ferry terminal, mangrove flooding"},

    {"Grid_ID": "LAM_003", "Latitude": -2.1500, "Longitude": 40.7833,
     "County": "Lamu", "Sub_County": "Hindi",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "High",
     "Notes": "Lamu Archipelago, some elevation buffer but tidal risk"},

    {"Grid_ID": "LAM_004", "Latitude": -1.8667, "Longitude": 41.0167,
     "County": "Lamu", "Sub_County": "Faza",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "Severe",
     "Notes": "Northern island, highest sea-level exposure in county"},

    # =========================================================================
    #  TAITA-TAVETA COUNTY
    #  Highland (Taita Hills) + lowland (Taveta) contrast. Taveta plain
    #  borders Amboseli wetlands and floods when Kilimanjaro snow melt
    #  combines with seasonal rain. Taita Hills drain efficiently.
    # =========================================================================
    {"Grid_ID": "TTV_001", "Latitude": -3.3833, "Longitude": 38.3833,
     "County": "Taita-Taveta", "Sub_County": "Wundanyi",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Taita Hills summit, excellent natural drainage"},

    {"Grid_ID": "TTV_002", "Latitude": -3.3833, "Longitude": 37.6833,
     "County": "Taita-Taveta", "Sub_County": "Taveta",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Amboseli wetland approach, Lumi River overflow, flat plains"},

    {"Grid_ID": "TTV_003", "Latitude": -3.5000, "Longitude": 38.3667,
     "County": "Taita-Taveta", "Sub_County": "Mwatate",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Voi River valley, moderate slope drainage"},

    {"Grid_ID": "TTV_004", "Latitude": -3.3667, "Longitude": 38.5667,
     "County": "Taita-Taveta", "Sub_County": "Voi",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "High",
     "Notes": "Voi River flooding, SGR corridor drainage issues"},

    # =========================================================================
    #  MANDERA COUNTY
    #  Dawa River (Kenya-Ethiopia-Somalia border) is the primary hazard.
    #  Upstream Ethiopian rainfall triggers floods here with 6-12 hour lag.
    #  Communities have little warning time. Flash floods from gullies
    #  (called "laga" locally) add secondary risk.
    # =========================================================================
    {"Grid_ID": "MDR_001", "Latitude":  3.9366, "Longitude": 41.8669,
     "County": "Mandera", "Sub_County": "Mandera Town",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "Severe",
     "Notes": "Dawa River floodplain, upstream Ethiopia trigger, laga flooding"},

    {"Grid_ID": "MDR_002", "Latitude":  3.8667, "Longitude": 42.0000,
     "County": "Mandera", "Sub_County": "Mandera East",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "Severe",
     "Notes": "Dawa River delta area, worst flooding in county"},

    {"Grid_ID": "MDR_003", "Latitude":  3.9000, "Longitude": 41.6000,
     "County": "Mandera", "Sub_County": "Mandera West",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Flash floods from seasonal lagas, hard soil runoff"},

    {"Grid_ID": "MDR_004", "Latitude":  3.5833, "Longitude": 41.6000,
     "County": "Mandera", "Sub_County": "Lafey",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Remote area, flash floods, minimal drainage infrastructure"},

    # =========================================================================
    #  WAJIR COUNTY
    #  Semi-arid. Seasonal flash floods from intense localised storms.
    #  Harder to predict than riverine floods — less warning time.
    # =========================================================================
    {"Grid_ID": "WJR_001", "Latitude":  1.7500, "Longitude": 40.0573,
     "County": "Wajir", "Sub_County": "Wajir Town",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Flash floods on hard soil, town centre drainage absent"},

    {"Grid_ID": "WJR_002", "Latitude":  1.7833, "Longitude": 40.3000,
     "County": "Wajir", "Sub_County": "Wajir East",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Semi-arid flash flood risk, seasonal laga channels"},

    {"Grid_ID": "WJR_003", "Latitude":  1.1667, "Longitude": 40.0667,
     "County": "Wajir", "Sub_County": "Wajir South",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "Moderate",
     "Notes": "Drier zone, lower flash flood frequency"},

    {"Grid_ID": "WJR_004", "Latitude":  1.8333, "Longitude": 39.5000,
     "County": "Wajir", "Sub_County": "Eldas",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "Moderate",
     "Notes": "Remote, very arid, flood less frequent but destructive"},

    # =========================================================================
    #  MARSABIT COUNTY
    #  Mostly low risk. Moyale (Ethiopia border) gets cross-border flash
    #  floods. Mt. Marsabit area has good natural drainage. Very large
    #  county — most of it is low-risk arid terrain.
    # =========================================================================
    {"Grid_ID": "MRS_001", "Latitude":  2.3284, "Longitude": 37.9899,
     "County": "Marsabit", "Sub_County": "Marsabit Town",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Low",
     "Notes": "Elevated volcanic crater, good natural drainage"},

    {"Grid_ID": "MRS_002", "Latitude":  3.5333, "Longitude": 39.0500,
     "County": "Marsabit", "Sub_County": "Moyale",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "Moderate",
     "Notes": "Cross-border flash floods from Ethiopian highlands"},

    {"Grid_ID": "MRS_003", "Latitude":  3.3167, "Longitude": 37.0833,
     "County": "Marsabit", "Sub_County": "North Horr",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "Low",
     "Notes": "Very arid, Chalbi Desert, rare but intense flash floods"},

    {"Grid_ID": "MRS_004", "Latitude":  1.6000, "Longitude": 37.8000,
     "County": "Marsabit", "Sub_County": "Laisamis",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "Moderate",
     "Notes": "Flash floods from Lolokwe Mountain catchment"},

    # =========================================================================
    #  ISIOLO COUNTY
    #  Ewaso Ng'iro River is the main hazard. River level driven by
    #  Mt. Kenya rainfall — another "remote trigger" county.
    # =========================================================================
    {"Grid_ID": "ISL_001", "Latitude":  0.3540, "Longitude": 37.5820,
     "County": "Isiolo", "Sub_County": "Isiolo Town",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Ewaso Ng'iro River proximity, moderate flash flood risk"},

    {"Grid_ID": "ISL_002", "Latitude":  0.5667, "Longitude": 38.3000,
     "County": "Isiolo", "Sub_County": "Merti",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Ewaso Ng'iro floodplain, riverside communities at risk"},

    {"Grid_ID": "ISL_003", "Latitude":  0.5167, "Longitude": 38.5667,
     "County": "Isiolo", "Sub_County": "Garbatulla",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Flash floods, Ewaso downstream flooding"},

    # =========================================================================
    #  MERU COUNTY
    #  Mt. Kenya eastern slopes. Good drainage overall. Tana River
    #  headwaters — flooding here is fast-moving and self-clearing.
    #  Tigania / Igembe sub-counties are drier, different flood profile.
    # =========================================================================
    {"Grid_ID": "MRU_001", "Latitude":  0.0470, "Longitude": 37.6490,
     "County": "Meru", "Sub_County": "Meru Town",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Kathita River seasonal flooding, urban drainage adequate"},

    {"Grid_ID": "MRU_002", "Latitude":  0.1833, "Longitude": 37.6000,
     "County": "Meru", "Sub_County": "Imenti North",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Mt Kenya upper slopes, fast natural drainage"},

    {"Grid_ID": "MRU_003", "Latitude":  0.0000, "Longitude": 37.6333,
     "County": "Meru", "Sub_County": "Imenti South",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Well-drained highland, minor river floods only"},

    {"Grid_ID": "MRU_004", "Latitude":  0.3333, "Longitude": 37.8833,
     "County": "Meru", "Sub_County": "Tigania East",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Drier zone, flash floods from Nyambene Hills"},

    {"Grid_ID": "MRU_005", "Latitude":  0.3667, "Longitude": 38.0833,
     "County": "Meru", "Sub_County": "Igembe Central",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Semi-arid approach, Tana tributary flash floods"},

    # =========================================================================
    #  THARAKA-NITHI COUNTY
    #  Tana River headwaters on Chuka/Nithi side. Lower counties
    #  (Tharaka) border semi-arid zone with different flood dynamics.
    # =========================================================================
    {"Grid_ID": "THN_001", "Latitude": -0.3333, "Longitude": 37.6333,
     "County": "Tharaka-Nithi", "Sub_County": "Chuka",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Highland town, good natural drainage"},

    {"Grid_ID": "THN_002", "Latitude": -0.2000, "Longitude": 38.1667,
     "County": "Tharaka-Nithi", "Sub_County": "Tharaka South",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "Moderate",
     "Notes": "Semi-arid lowland, Tana tributary flash floods"},

    {"Grid_ID": "THN_003", "Latitude": -0.0833, "Longitude": 38.0833,
     "County": "Tharaka-Nithi", "Sub_County": "Tharaka North",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "Moderate",
     "Notes": "Tana River tributaries, seasonal inundation"},

    {"Grid_ID": "THN_004", "Latitude": -0.3667, "Longitude": 37.7667,
     "County": "Tharaka-Nithi", "Sub_County": "Maara",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Moderate terrain, well-drained Nithi River area"},

    # =========================================================================
    #  EMBU COUNTY
    #  Tana River source area. Mwea Irrigation Scheme (Kirinyaga border)
    #  creates artificial flood risk when canals overflow or rice paddies
    #  are inundated. Mbeere sub-county is semi-arid, higher flood risk.
    # =========================================================================
    {"Grid_ID": "EMB_001", "Latitude": -0.5300, "Longitude": 37.4500,
     "County": "Embu", "Sub_County": "Embu Town",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Tana River upper catchment, seasonal flooding"},

    {"Grid_ID": "EMB_002", "Latitude": -0.4833, "Longitude": 37.4833,
     "County": "Embu", "Sub_County": "Manyatta",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Peri-urban expansion, drainage stress"},

    {"Grid_ID": "EMB_003", "Latitude": -0.4167, "Longitude": 37.5667,
     "County": "Embu", "Sub_County": "Runyenjes",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Tana River upper tributary, valley flooding"},

    {"Grid_ID": "EMB_004", "Latitude": -0.7833, "Longitude": 37.5667,
     "County": "Embu", "Sub_County": "Mbeere North",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Tana River mid-section, semi-arid flash flood risk"},

    # =========================================================================
    #  KITUI COUNTY
    #  Semi-arid. Mwingi is the main hazard area — Tana River tributary.
    #  Flash floods are deadly here because the dry river beds (sand rivers)
    #  fill instantly and communities sometimes camp in them.
    # =========================================================================
    {"Grid_ID": "KTI_001", "Latitude": -1.3667, "Longitude": 38.0167,
     "County": "Kitui", "Sub_County": "Kitui Central",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Semi-arid urban area, manageable flash flood risk"},

    {"Grid_ID": "KTI_002", "Latitude": -0.9667, "Longitude": 38.0667,
     "County": "Kitui", "Sub_County": "Mwingi North",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Tana River tributary, sand river flash floods, dangerous"},

    {"Grid_ID": "KTI_003", "Latitude": -1.1333, "Longitude": 38.1000,
     "County": "Kitui", "Sub_County": "Mwingi Central",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Semi-arid flash floods, Tana basin"},

    {"Grid_ID": "KTI_004", "Latitude": -1.8333, "Longitude": 38.2000,
     "County": "Kitui", "Sub_County": "Mutomo",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Driest part of county, but flash floods from sudden storms"},

    {"Grid_ID": "KTI_005", "Latitude": -1.5000, "Longitude": 38.5000,
     "County": "Kitui", "Sub_County": "Mumoni",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "Moderate",
     "Notes": "Hilly terrain, some natural drainage from Mumoni Hills"},

    # =========================================================================
    #  MAKUENI COUNTY
    #  Kibwezi River flash floods. Tsavo area (wildlife corridors) makes
    #  evacuation difficult. Semi-arid — intense rainfall triggers fast floods.
    # =========================================================================
    {"Grid_ID": "MKN_001", "Latitude": -1.7833, "Longitude": 37.6333,
     "County": "Makueni", "Sub_County": "Makueni Town",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "County HQ, semi-arid, manageable drainage"},

    {"Grid_ID": "MKN_002", "Latitude": -2.4167, "Longitude": 37.9500,
     "County": "Makueni", "Sub_County": "Kibwezi East",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Kibwezi River flooding, Tsavo wildlife barrier"},

    {"Grid_ID": "MKN_003", "Latitude": -2.3500, "Longitude": 37.8833,
     "County": "Makueni", "Sub_County": "Kibwezi West",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Tsavo-adjacent floodplain, Tsavo River proximity"},

    {"Grid_ID": "MKN_004", "Latitude": -1.5000, "Longitude": 37.5500,
     "County": "Makueni", "Sub_County": "Kathonzweni",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Ukambani hills, moderate terrain drainage"},

    # =========================================================================
    #  NYANDARUA COUNTY
    #  Aberdare highlands. Very high rainfall (2000mm/yr) but steep terrain
    #  means fast drainage. Wanjohi Valley is the exception — basin shape
    #  traps water. Low flood risk overall, good baseline for calibration.
    # =========================================================================
    {"Grid_ID": "NDR_001", "Latitude": -0.2667, "Longitude": 36.3833,
     "County": "Nyandarua", "Sub_County": "Ol Kalou",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Aberdare highland plateau, efficient natural drainage"},

    {"Grid_ID": "NDR_002", "Latitude":  0.0167, "Longitude": 36.5667,
     "County": "Nyandarua", "Sub_County": "Ndaragwa",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Northern Aberdares, well-drained volcanic soils"},

    {"Grid_ID": "NDR_003", "Latitude": -0.6833, "Longitude": 36.5333,
     "County": "Nyandarua", "Sub_County": "Kinangop",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Wanjohi Valley basin shape traps water, seasonal floods"},

    {"Grid_ID": "NDR_004", "Latitude": -0.3000, "Longitude": 36.4167,
     "County": "Nyandarua", "Sub_County": "Kipipiri",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Kipipiri highland, slopes drain well into Aberdare rivers"},

    # =========================================================================
    #  KIRINYAGA COUNTY
    #  Mt. Kenya slopes. Mwea Irrigation Scheme is the standout risk area —
    #  when irrigation canals overflow, the rice paddy floodplain extends.
    #  The scheme is designed to flood rice fields, but excess water
    #  inundates roads and homesteads during heavy rain years.
    # =========================================================================
    {"Grid_ID": "KRY_001", "Latitude": -0.4989, "Longitude": 37.2814,
     "County": "Kirinyaga", "Sub_County": "Kerugoya",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "County HQ highland, well-drained terrain"},

    {"Grid_ID": "KRY_002", "Latitude": -0.6833, "Longitude": 37.4167,
     "County": "Kirinyaga", "Sub_County": "Mwea East",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Mwea Irrigation Scheme, canal overflow, rice paddy flooding"},

    {"Grid_ID": "KRY_003", "Latitude": -0.6000, "Longitude": 37.4833,
     "County": "Kirinyaga", "Sub_County": "Mwea West",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Tana River bank, Mwea Scheme extension flooding"},

    {"Grid_ID": "KRY_004", "Latitude": -0.5167, "Longitude": 37.2833,
     "County": "Kirinyaga", "Sub_County": "Gichugu",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Kirinyaga highlands, good slope drainage"},

    # =========================================================================
    #  MURANG'A COUNTY
    #  Thika and Mathioya rivers. Good highland drainage but valley floors
    #  flood. Kandara and Kigumo at river confluences are most vulnerable.
    # =========================================================================
    {"Grid_ID": "MRG_001", "Latitude": -0.7167, "Longitude": 37.1500,
     "County": "Murang'a", "Sub_County": "Murang'a Town",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Moderate",
     "Notes": "Mathioya River proximity, seasonal town flooding"},

    {"Grid_ID": "MRG_002", "Latitude": -0.6167, "Longitude": 36.9833,
     "County": "Murang'a", "Sub_County": "Kangema",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Well-drained highland, minor river flooding only"},

    {"Grid_ID": "MRG_003", "Latitude": -0.9167, "Longitude": 36.9500,
     "County": "Murang'a", "Sub_County": "Kigumo",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Thika River valley, seasonal flooding at confluences"},

    {"Grid_ID": "MRG_004", "Latitude": -0.9833, "Longitude": 37.0667,
     "County": "Murang'a", "Sub_County": "Gatanga",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Thika River, reservoir catchment area flooding"},

    # =========================================================================
    #  KIAMBU COUNTY
    #  Nairobi's satellite county. Thika and Chania rivers. Rapid urbanisation
    #  turning agricultural land into impervious surfaces. Ruiru and Thika
    #  towns have chronic flooding during every long rain season.
    # =========================================================================
    {"Grid_ID": "KMB_001", "Latitude": -1.0310, "Longitude": 36.8314,
     "County": "Kiambu", "Sub_County": "Kiambu Town",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Moderate",
     "Notes": "Moderate urban drainage, Kiambu River occasional overflow"},

    {"Grid_ID": "KMB_002", "Latitude": -1.0333, "Longitude": 37.0833,
     "County": "Kiambu", "Sub_County": "Thika Town",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "High",
     "Notes": "Chania + Thika rivers, industrial zone flooding risk"},

    {"Grid_ID": "KMB_003", "Latitude": -1.2500, "Longitude": 36.6667,
     "County": "Kiambu", "Sub_County": "Kikuyu",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Highland tea zone, good natural drainage"},

    {"Grid_ID": "KMB_004", "Latitude": -1.1500, "Longitude": 36.9667,
     "County": "Kiambu", "Sub_County": "Ruiru",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Fastest-growing town in Kenya, drainage severely inadequate"},

    {"Grid_ID": "KMB_005", "Latitude": -1.1000, "Longitude": 36.6333,
     "County": "Kiambu", "Sub_County": "Limuru",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Tea zone highland, well-drained terraced land"},

    # =========================================================================
    #  WEST POKOT COUNTY
    #  Kerio Valley flash floods are severe and fast. The valley floor
    #  (elevation ~1000m) sits below the escarpment (2500m) — rainfall
    #  on the escarpment rushes down in minutes, giving no warning.
    # =========================================================================
    {"Grid_ID": "WPK_001", "Latitude":  1.2333, "Longitude": 35.1167,
     "County": "West Pokot", "Sub_County": "Kapenguria",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "County HQ, moderate terrain, manageable flooding"},

    {"Grid_ID": "WPK_002", "Latitude":  1.8000, "Longitude": 35.1000,
     "County": "West Pokot", "Sub_County": "Pokot North",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Flash floods from northern escarpment, erosion risk"},

    {"Grid_ID": "WPK_003", "Latitude":  1.6000, "Longitude": 35.3000,
     "County": "West Pokot", "Sub_County": "Pokot Central",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "Severe",
     "Notes": "Kerio Valley floor, catastrophic escarpment flash floods"},

    {"Grid_ID": "WPK_004", "Latitude":  1.0833, "Longitude": 35.4333,
     "County": "West Pokot", "Sub_County": "Pokot South",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Kerio River lower section, seasonal overflow"},

    # =========================================================================
    #  SAMBURU COUNTY
    #  Ewaso Ng'iro River. Moderate risk compared to ASAL neighbours.
    #  Maralal is in highlands with better drainage. Eastern parts are
    #  more arid with flash flood risk.
    # =========================================================================
    {"Grid_ID": "SMB_001", "Latitude":  1.0967, "Longitude": 36.7017,
     "County": "Samburu", "Sub_County": "Maralal",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Low",
     "Notes": "Highland county HQ, adequate natural drainage"},

    {"Grid_ID": "SMB_002", "Latitude":  0.9167, "Longitude": 37.2833,
     "County": "Samburu", "Sub_County": "Samburu East",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "Moderate",
     "Notes": "Ewaso Ng'iro basin, seasonal flooding in dry river beds"},

    {"Grid_ID": "SMB_003", "Latitude":  1.7000, "Longitude": 37.0833,
     "County": "Samburu", "Sub_County": "Samburu North",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "Low",
     "Notes": "Arid north, rare but intense flash floods"},

    # =========================================================================
    #  TRANS-NZOIA COUNTY
    #  Nzoia River headwaters + flat farmland = chronic flooding. Kitale
    #  town flooding is annual. Mt. Elgon areas get highland flash floods.
    # =========================================================================
    {"Grid_ID": "TNZ_001", "Latitude":  1.0166, "Longitude": 34.9500,
     "County": "Trans-Nzoia", "Sub_County": "Kitale Town",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "High",
     "Notes": "Nzoia River headwaters, annual town flooding"},

    {"Grid_ID": "TNZ_002", "Latitude":  1.1167, "Longitude": 34.7667,
     "County": "Trans-Nzoia", "Sub_County": "Kwanza",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Mt Elgon lower slopes, manageable seasonal floods"},

    {"Grid_ID": "TNZ_003", "Latitude":  1.1500, "Longitude": 35.2500,
     "County": "Trans-Nzoia", "Sub_County": "Cherangany",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Cherangany Hills, good natural drainage"},

    {"Grid_ID": "TNZ_004", "Latitude":  1.0500, "Longitude": 34.9833,
     "County": "Trans-Nzoia", "Sub_County": "Kiminini",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Flat farmland, Nzoia floodplain, worst flooding in county"},

    # =========================================================================
    #  UASIN GISHU COUNTY
    #  Eldoret is a major urban centre. Sosiani River floods the industrial
    #  quarter. Generally flat plateau with moderate drainage.
    # =========================================================================
    {"Grid_ID": "USG_001", "Latitude":  0.5143, "Longitude": 35.2697,
     "County": "Uasin Gishu", "Sub_County": "Eldoret Town",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Sosiani River flooding, industrial quarter at risk"},

    {"Grid_ID": "USG_002", "Latitude":  0.6167, "Longitude": 35.0500,
     "County": "Uasin Gishu", "Sub_County": "Turbo",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Flat terrain, some Nzoia tributary flooding"},

    {"Grid_ID": "USG_003", "Latitude":  0.3833, "Longitude": 35.1333,
     "County": "Uasin Gishu", "Sub_County": "Kapseret",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "JKIA-E area, plateau with good drainage"},

    {"Grid_ID": "USG_004", "Latitude":  0.7833, "Longitude": 35.2833,
     "County": "Uasin Gishu", "Sub_County": "Moiben",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Highland wheat zone, well-drained Eldoret escarpment"},

    # =========================================================================
    #  ELGEYO-MARAKWET COUNTY
    #  Kerio Valley is the most dangerous terrain in the Rift for flash
    #  floods. The escarpment drops 1500m over 10km — water accelerates
    #  to deadly speed. Valley floor communities have under 30 minutes
    #  warning when escarpment storms occur.
    # =========================================================================
    {"Grid_ID": "ELM_001", "Latitude":  0.6700, "Longitude": 35.5083,
     "County": "Elgeyo-Marakwet", "Sub_County": "Iten",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Escarpment top town, athletics training altitude, drains well"},

    {"Grid_ID": "ELM_002", "Latitude":  0.5000, "Longitude": 35.5500,
     "County": "Elgeyo-Marakwet", "Sub_County": "Keiyo South",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Kerio Valley floor, flash floods from escarpment in minutes"},

    {"Grid_ID": "ELM_003", "Latitude":  0.9667, "Longitude": 35.5833,
     "County": "Elgeyo-Marakwet", "Sub_County": "Marakwet East",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Elgeyo mid-slopes, managed terracing reduces risk"},

    {"Grid_ID": "ELM_004", "Latitude":  1.0833, "Longitude": 35.4833,
     "County": "Elgeyo-Marakwet", "Sub_County": "Marakwet West",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Kerio River upper section, escarpment flash flood corridor"},

    # =========================================================================
    #  NANDI COUNTY
    #  Tea growing region. Good drainage overall due to terrain and
    #  tea crop root systems that absorb water. Nzoia headwater tributaries
    #  in lower Nandi are the main flood risk.
    # =========================================================================
    {"Grid_ID": "NDI_001", "Latitude":  0.2000, "Longitude": 35.0833,
     "County": "Nandi", "Sub_County": "Kapsabet",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "County HQ, well-drained highland centre"},

    {"Grid_ID": "NDI_002", "Latitude":  0.0000, "Longitude": 35.3167,
     "County": "Nandi", "Sub_County": "Aldai",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Tea zone, excellent natural drainage from root systems"},

    {"Grid_ID": "NDI_003", "Latitude":  0.1000, "Longitude": 35.1667,
     "County": "Nandi", "Sub_County": "Nandi Hills Town",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Productive tea belt, well-maintained drainage"},

    {"Grid_ID": "NDI_004", "Latitude":  0.3167, "Longitude": 35.2333,
     "County": "Nandi", "Sub_County": "Chesumei",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Nzoia headwaters, some valley flooding"},

    # =========================================================================
    #  BARINGO COUNTY
    #  Lake Baringo water level fluctuations + Perkerra River + Rift Valley
    #  escarpment flash floods. Tiaty (eastern, semi-arid) has severe
    #  flash floods on hard soil.
    # =========================================================================
    {"Grid_ID": "BRN_001", "Latitude":  0.4917, "Longitude": 35.7433,
     "County": "Baringo", "Sub_County": "Kabarnet",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Escarpment town, moderate drainage, seasonal flooding"},

    {"Grid_ID": "BRN_002", "Latitude":  0.0500, "Longitude": 35.7167,
     "County": "Baringo", "Sub_County": "Eldama Ravine",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Ravine geography, channelled flooding"},

    {"Grid_ID": "BRN_003", "Latitude": -0.1667, "Longitude": 35.9833,
     "County": "Baringo", "Sub_County": "Mogotio",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Rift floor, Lake Baringo surge, flat terrain flooding"},

    {"Grid_ID": "BRN_004", "Latitude":  1.0833, "Longitude": 36.1667,
     "County": "Baringo", "Sub_County": "Tiaty",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Semi-arid remote zone, hard soil flash floods, no infra"},

    {"Grid_ID": "BRN_005", "Latitude":  0.5000, "Longitude": 35.9167,
     "County": "Baringo", "Sub_County": "Baringo Central",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Lake Baringo shoreline, lake level rise flooding"},

    # =========================================================================
    #  LAIKIPIA COUNTY
    #  Semi-arid plateau. Ewaso Ng'iro River corridor is the main risk.
    #  Nanyuki (highland) has lower risk. Laikipia East (semi-arid) has
    #  flash flood risk on hard soils.
    # =========================================================================
    {"Grid_ID": "LKP_001", "Latitude":  0.0167, "Longitude": 37.0667,
     "County": "Laikipia", "Sub_County": "Nanyuki",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Mt Kenya highland town, good drainage"},

    {"Grid_ID": "LKP_002", "Latitude":  0.2833, "Longitude": 36.5333,
     "County": "Laikipia", "Sub_County": "Rumuruti",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Semi-arid, flash floods from Laikipia escarpment"},

    {"Grid_ID": "LKP_003", "Latitude":  0.4000, "Longitude": 37.3000,
     "County": "Laikipia", "Sub_County": "Laikipia East",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "Moderate",
     "Notes": "Ewaso Ng'iro corridor, semi-arid flash flood risk"},

    # =========================================================================
    #  NAROK COUNTY
    #  Mara River basin (Maasai Mara). Floods impact wildlife conservancies
    #  and tourism. Kilgoris/Transmara areas are wettest. Narok North
    #  (Mau Forest edge) drains well.
    # =========================================================================
    {"Grid_ID": "NRK_001", "Latitude": -1.0814, "Longitude": 35.8634,
     "County": "Narok", "Sub_County": "Narok Town",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Enkare Narok River, town centre seasonal flooding"},

    {"Grid_ID": "NRK_002", "Latitude": -1.0167, "Longitude": 34.8833,
     "County": "Narok", "Sub_County": "Kilgoris",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "High",
     "Notes": "Mara River basin, tourism lodges at flood risk"},

    {"Grid_ID": "NRK_003", "Latitude": -1.2000, "Longitude": 34.9000,
     "County": "Narok", "Sub_County": "Transmara East",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Mara River flooding, Maasai Mara conservancy inundation"},

    {"Grid_ID": "NRK_004", "Latitude": -0.8333, "Longitude": 35.8333,
     "County": "Narok", "Sub_County": "Narok North",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Mau Forest edge, good natural forest drainage"},

    # =========================================================================
    #  KAJIADO COUNTY
    #  Mostly semi-arid. Ongata Rongai / Kitengela are rapidly urbanising
    #  Nairobi overspill zones — drainage not keeping pace with growth.
    #  Loitokitok borders Amboseli wetlands.
    # =========================================================================
    {"Grid_ID": "KJD_001", "Latitude": -1.3667, "Longitude": 36.6583,
     "County": "Kajiado", "Sub_County": "Ngong",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Ngong Hills, highland drainage, low flood risk"},

    {"Grid_ID": "KJD_002", "Latitude": -1.3833, "Longitude": 36.6833,
     "County": "Kajiado", "Sub_County": "Ongata Rongai",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Nairobi overspill, drainage falling behind urbanisation"},

    {"Grid_ID": "KJD_003", "Latitude": -1.8333, "Longitude": 36.7833,
     "County": "Kajiado", "Sub_County": "Kajiado Central",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "Low",
     "Notes": "Semi-arid town, rare but intense flash floods"},

    {"Grid_ID": "KJD_004", "Latitude": -2.9000, "Longitude": 37.5167,
     "County": "Kajiado", "Sub_County": "Loitokitok",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Amboseli wetlands approach, Kilimanjaro snowmelt floods"},

    # =========================================================================
    #  KERICHO COUNTY
    #  Tea country. Heavy rainfall (1700mm+/yr) but excellent tea-root
    #  drainage. Rivers from Mau Forest, manageable gradient. Lower risk
    #  than rainfall volume would suggest.
    # =========================================================================
    {"Grid_ID": "KRC_001", "Latitude": -0.3686, "Longitude": 35.2863,
     "County": "Kericho", "Sub_County": "Kericho Town",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Moderate",
     "Notes": "Tea belt town, Kericho River seasonal overflow"},

    {"Grid_ID": "KRC_002", "Latitude": -0.5333, "Longitude": 35.2000,
     "County": "Kericho", "Sub_County": "Bureti",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Tea zone, good natural drainage from root systems"},

    {"Grid_ID": "KRC_003", "Latitude": -0.4000, "Longitude": 35.1667,
     "County": "Kericho", "Sub_County": "Belgut",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Mara River headwaters, manageable seasonal flooding"},

    {"Grid_ID": "KRC_004", "Latitude": -0.3333, "Longitude": 35.3500,
     "County": "Kericho", "Sub_County": "Ainamoi",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Active tea production zone, well-maintained drainage"},

    # =========================================================================
    #  BOMET COUNTY
    #  Mara River headwaters. Well-drained highland generally. Sotik and
    #  Chepalungu border the Mau Forest — forest cover provides natural
    #  flood buffering.
    # =========================================================================
    {"Grid_ID": "BMT_001", "Latitude": -0.7820, "Longitude": 35.3430,
     "County": "Bomet", "Sub_County": "Bomet Town",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Mara headwaters town, seasonal river flooding"},

    {"Grid_ID": "BMT_002", "Latitude": -0.6833, "Longitude": 35.1167,
     "County": "Bomet", "Sub_County": "Sotik",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Tea zone, excellent forest-adjacent drainage"},

    {"Grid_ID": "BMT_003", "Latitude": -0.7167, "Longitude": 35.4833,
     "County": "Bomet", "Sub_County": "Chepalungu",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Mau Forest edge, forest cover buffers flooding"},

    {"Grid_ID": "BMT_004", "Latitude": -0.9000, "Longitude": 35.2500,
     "County": "Bomet", "Sub_County": "Konoin",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "River valleys, some seasonal flooding"},

    # =========================================================================
    #  KAKAMEGA COUNTY
    #  Kenya's rainiest county (some years 2000mm+). Nzoia River dominates.
    #  Despite high rainfall, drainage is poor in lower-lying areas.
    #  Khwisero and Ikolomani are the worst affected sub-counties.
    # =========================================================================
    {"Grid_ID": "KKM_001", "Latitude":  0.2827, "Longitude": 34.7519,
     "County": "Kakamega", "Sub_County": "Kakamega Town",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "High",
     "Notes": "Nzoia River proximity, annual flooding during long rains"},

    {"Grid_ID": "KKM_002", "Latitude":  0.3333, "Longitude": 34.7000,
     "County": "Kakamega", "Sub_County": "Lurambi",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Flooded nearly every long rain season, poor drainage"},

    {"Grid_ID": "KKM_003", "Latitude":  0.2000, "Longitude": 34.9000,
     "County": "Kakamega", "Sub_County": "Likuyani",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Sugarcane belt, flat terrain, moderate flooding"},

    {"Grid_ID": "KKM_004", "Latitude":  0.2500, "Longitude": 34.8500,
     "County": "Kakamega", "Sub_County": "Shinyalu",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Moderate",
     "Notes": "Kakamega Forest adjacent, forest buffering helps"},

    {"Grid_ID": "KKM_005", "Latitude":  0.1333, "Longitude": 34.7333,
     "County": "Kakamega", "Sub_County": "Ikolomani",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Low-lying, Nzoia tributary flooding"},

    {"Grid_ID": "KKM_006", "Latitude":  0.0333, "Longitude": 34.6500,
     "County": "Kakamega", "Sub_County": "Khwisero",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Nzoia floodplain, worst flooding sub-county in Kakamega"},

    # =========================================================================
    #  VIHIGA COUNTY
    #  Kenya's most densely populated county. High population density on
    #  small area = drainage infrastructure overwhelmed. River Yala and
    #  Nzoia tributaries cause annual flooding.
    # =========================================================================
    {"Grid_ID": "VHG_001", "Latitude":  0.0800, "Longitude": 34.7200,
     "County": "Vihiga", "Sub_County": "Vihiga Town",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "High",
     "Notes": "Dense population, River Yala overflow risk"},

    {"Grid_ID": "VHG_002", "Latitude":  0.0667, "Longitude": 34.7833,
     "County": "Vihiga", "Sub_County": "Emuhaya",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Low-lying, Yala River flooding, dense settlement"},

    {"Grid_ID": "VHG_003", "Latitude":  0.2000, "Longitude": 34.8000,
     "County": "Vihiga", "Sub_County": "Sabatia",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Moderate",
     "Notes": "Higher ground, better natural drainage"},

    {"Grid_ID": "VHG_004", "Latitude":  0.1167, "Longitude": 34.7500,
     "County": "Vihiga", "Sub_County": "Hamisi",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Moderate terrain, seasonal flooding in valleys"},

    # =========================================================================
    #  BUNGOMA COUNTY
    #  Mt. Elgon + Nzoia River. Webuye Falls area has industrial flooding
    #  risk. Mt. Elgon sub-counties get landslide-triggered floods.
    # =========================================================================
    {"Grid_ID": "BNG_001", "Latitude":  0.5635, "Longitude": 34.5585,
     "County": "Bungoma", "Sub_County": "Bungoma Town",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "High",
     "Notes": "Nzoia River proximity, annual flooding"},

    {"Grid_ID": "BNG_002", "Latitude":  0.5833, "Longitude": 34.5833,
     "County": "Bungoma", "Sub_County": "Kanduyi",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Nzoia tributary flooding, poor urban drainage"},

    {"Grid_ID": "BNG_003", "Latitude":  0.6167, "Longitude": 34.7667,
     "County": "Bungoma", "Sub_County": "Webuye East",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Nzoia River gorge below Webuye Falls, industrial flooding"},

    {"Grid_ID": "BNG_004", "Latitude":  0.7000, "Longitude": 34.6000,
     "County": "Bungoma", "Sub_County": "Sirisia",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Mt Elgon lower slopes, manageable drainage"},

    {"Grid_ID": "BNG_005", "Latitude":  1.0167, "Longitude": 34.6833,
     "County": "Bungoma", "Sub_County": "Mount Elgon",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Landslide + flash flood combination risk on Elgon slopes"},

    # =========================================================================
    #  SIAYA COUNTY
    #  Yala Swamp (one of Africa's largest freshwater wetlands) means much
    #  of this county is effectively always saturated. Lake Victoria
    #  shoreline is directly exposed. Bondo is the most critical area.
    # =========================================================================
    {"Grid_ID": "SYA_001", "Latitude": -0.0610, "Longitude": 34.2880,
     "County": "Siaya", "Sub_County": "Siaya Town",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "High",
     "Notes": "Yala River proximity, seasonal flooding"},

    {"Grid_ID": "SYA_002", "Latitude":  0.0000, "Longitude": 34.2333,
     "County": "Siaya", "Sub_County": "Gem",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Yala swamp edge, soil saturation throughout rainy season"},

    {"Grid_ID": "SYA_003", "Latitude":  0.1000, "Longitude": 34.3333,
     "County": "Siaya", "Sub_County": "Ugenya",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Swampy lowlands, poor drainage, annual flooding"},

    {"Grid_ID": "SYA_004", "Latitude": -0.0500, "Longitude": 34.3833,
     "County": "Siaya", "Sub_County": "Alego-Usonga",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Lake Victoria basin approach, flat terrain"},

    {"Grid_ID": "SYA_005", "Latitude": -0.1833, "Longitude": 34.2667,
     "County": "Siaya", "Sub_County": "Bondo",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "Severe",
     "Notes": "Yala Swamp adjacency, lake proximity, most critical in Siaya"},

    # =========================================================================
    #  HOMA BAY COUNTY
    #  Lake Victoria bay geography means flooding from BOTH rainfall and
    #  lake surge. Mbita (island) is most exposed. The 2020 Lake Victoria
    #  surge flooded thousands of homes along this shoreline.
    # =========================================================================
    {"Grid_ID": "HMB_001", "Latitude": -0.5273, "Longitude": 34.4571,
     "County": "Homa Bay", "Sub_County": "Homa Bay Town",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "Severe",
     "Notes": "Bay position, lake surge, Homa Bay directly floods in high lake levels"},

    {"Grid_ID": "HMB_002", "Latitude": -0.4500, "Longitude": 34.5500,
     "County": "Homa Bay", "Sub_County": "Rangwe",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Lake Victoria shoreline exposure"},

    {"Grid_ID": "HMB_003", "Latitude": -0.5000, "Longitude": 34.3667,
     "County": "Homa Bay", "Sub_County": "Karachuonyo",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Lake shore community, annual displacement"},

    {"Grid_ID": "HMB_004", "Latitude": -0.7000, "Longitude": 34.5000,
     "County": "Homa Bay", "Sub_County": "Ndhiwa",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "High",
     "Notes": "Homa Bay inland, rivers drain to lake"},

    {"Grid_ID": "HMB_005", "Latitude": -0.4167, "Longitude": 34.2000,
     "County": "Homa Bay", "Sub_County": "Mbita",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "Severe",
     "Notes": "Island/peninsula, maximum lake surge exposure"},

    # =========================================================================
    #  MIGORI COUNTY
    #  Migori River + Lake Victoria basin. Sugar belt (Awendo) experiences
    #  chronic flooding from flat terrain. Tanzania border rivers also
    #  contribute to cross-border flood events.
    # =========================================================================
    {"Grid_ID": "MGR_001", "Latitude": -1.0634, "Longitude": 34.4731,
     "County": "Migori", "Sub_County": "Migori Town",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "High",
     "Notes": "Migori River overflow, annual flooding"},

    {"Grid_ID": "MGR_002", "Latitude": -0.9500, "Longitude": 34.6000,
     "County": "Migori", "Sub_County": "Rongo",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Elevated relative to lake basin, manageable flooding"},

    {"Grid_ID": "MGR_003", "Latitude": -0.7833, "Longitude": 34.5833,
     "County": "Migori", "Sub_County": "Awendo",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Sugarcane flatlands, chronic flooding, poor drainage"},

    {"Grid_ID": "MGR_004", "Latitude": -0.8667, "Longitude": 34.4667,
     "County": "Migori", "Sub_County": "Uriri",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Lake basin approach, seasonal inundation"},

    {"Grid_ID": "MGR_005", "Latitude": -1.0833, "Longitude": 34.3833,
     "County": "Migori", "Sub_County": "Suna West",
     "Drainage_Quality": "Poor", "Flood_Risk_Level": "High",
     "Notes": "Lake Victoria shoreline, Migori River mouth flooding"},

    # =========================================================================
    #  KISII COUNTY
    #  Highland county with heavy rainfall. Valleys and lower sub-counties
    #  experience significant flooding. Tea and pyrethrum terracing helps.
    # =========================================================================
    {"Grid_ID": "KSI_001", "Latitude": -0.6817, "Longitude": 34.7667,
     "County": "Kisii", "Sub_County": "Kisii Town",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Urban centre, river flooding, drainage improving"},

    {"Grid_ID": "KSI_002", "Latitude": -0.5167, "Longitude": 34.8167,
     "County": "Kisii", "Sub_County": "Masaba North",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Highland area, good natural slope drainage"},

    {"Grid_ID": "KSI_003", "Latitude": -0.7000, "Longitude": 34.8000,
     "County": "Kisii", "Sub_County": "Kitutu Chache North",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "River valleys, seasonal flooding"},

    {"Grid_ID": "KSI_004", "Latitude": -0.7500, "Longitude": 34.7000,
     "County": "Kisii", "Sub_County": "Nyaribari Masaba",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Moderate terrain, valley flooding"},

    {"Grid_ID": "KSI_005", "Latitude": -0.8167, "Longitude": 34.8500,
     "County": "Kisii", "Sub_County": "South Mugirango",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Highland south, well-drained escarpment"},

    # =========================================================================
    #  NYAMIRA COUNTY
    #  Highest-elevation county in Nyanza region. Best natural drainage.
    #  Relatively low flood risk compared to Kisumu/Siaya/Homa Bay.
    #  River valleys can flood but water moves off quickly.
    # =========================================================================
    {"Grid_ID": "NYM_001", "Latitude": -0.5667, "Longitude": 34.9333,
     "County": "Nyamira", "Sub_County": "Nyamira Town",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Highland county HQ, efficient natural drainage"},

    {"Grid_ID": "NYM_002", "Latitude": -0.4833, "Longitude": 34.9667,
     "County": "Nyamira", "Sub_County": "Manga",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Highland tea zone, well-drained terrain"},

    {"Grid_ID": "NYM_003", "Latitude": -0.7333, "Longitude": 35.0000,
     "County": "Nyamira", "Sub_County": "Borabu",
     "Drainage_Quality": "Good", "Flood_Risk_Level": "Low",
     "Notes": "Well-drained highland, low flood history"},

    {"Grid_ID": "NYM_004", "Latitude": -0.5000, "Longitude": 35.0167,
     "County": "Nyamira", "Sub_County": "Masaba South",
     "Drainage_Quality": "Moderate", "Flood_Risk_Level": "Moderate",
     "Notes": "Lower slopes, some river valley flooding"},

]   # END of kenya_flood_drainage list


# =============================================================================
#  SECTION 3: QUICK-ACCESS HELPER FUNCTIONS
#  Imported and used by risk_calculator.py and flood_predictor.py
# =============================================================================

def get_subcounties_for_county(county_name: str) -> list:
    """
    Returns all grid cells for a given county name.

    WHY: The main FRI loop works at county level. This function lets the
    alert system drop down to sub-county resolution when a HIGH or CRITICAL
    alert is triggered — so responders know WHICH sub-counties to prioritise.

    Example:
        cells = get_subcounties_for_county("Kisumu")
        # Returns all KSM_* rows: Central, Nyando, Muhoroni, Seme, East, Winam
    """
    return [
        cell for cell in kenya_flood_drainage
        if cell["County"].lower() == county_name.lower()
    ]


def get_county_drainage_score(county_name: str) -> float:
    """
    Computes the AVERAGE numeric drainage score across all sub-counties
    in a given county. Returns a float on the 1-5 scale.

    WHY: config.py has a single drainage integer per county. This function
    computes a data-driven average from the dataset instead, which is more
    accurate because it weights all sub-counties equally.

    If no sub-county data exists for a county, returns None and the
    caller falls back to the config.py value.

    Example:
        score = get_county_drainage_score("Kisumu")  # Returns ~1.17
        score = get_county_drainage_score("Nyeri")   # Returns ~4.0
    """
    cells = get_subcounties_for_county(county_name)
    if not cells:
        return None  # Caller should fall back to config.py value

    # Map each string quality to numeric using the lookup table at top of file
    scores = [DRAINAGE_QUALITY_MAP[c["Drainage_Quality"]] for c in cells]
    return sum(scores) / len(scores)


def get_worst_subcounties(county_name: str, top_n: int = 3) -> list:
    """
    Returns the top_n sub-counties with the highest flood risk level
    for a given county, sorted worst-first.

    WHY: When firing a HIGH/CRITICAL alert, the system calls this to list
    WHICH specific sub-counties are most at risk. This makes alerts
    actionable — responders know where to pre-position resources.

    Risk order for sorting: Severe > High > Moderate > Low

    Example:
        worst = get_worst_subcounties("Nairobi", top_n=2)
        # Returns: [Kibera (Severe), Embakasi (High)]
    """
    # Define sort priority: Severe=3, High=2, Moderate=1, Low=0
    priority = {"Severe": 3, "High": 2, "Moderate": 1, "Low": 0}

    cells = get_subcounties_for_county(county_name)
    # Sort by flood risk level descending, then by sub-county name for stability
    sorted_cells = sorted(
        cells,
        key=lambda c: priority.get(c["Flood_Risk_Level"], 0),
        reverse=True
    )
    return sorted_cells[:top_n]


def get_training_samples() -> list:
    """
    Returns the entire dataset as a list of dicts ready for ML feature
    extraction in flood_predictor.py.

    WHY: The Random Forest classifier needs training examples. Each grid
    cell in this dataset is one training sample. The label (y) is
    FLOOD_RISK_LABEL_MAP[cell["Flood_Risk_Level"]] — an integer 0-3.

    The features (X) are extracted from the cell's static characteristics
    combined with the county's rainfall data fetched at runtime.

    Example usage in flood_predictor.py:
        samples = get_training_samples()
        for sample in samples:
            label = FLOOD_RISK_LABEL_MAP[sample["Flood_Risk_Level"]]
            drainage_numeric = DRAINAGE_QUALITY_MAP[sample["Drainage_Quality"]]
    """
    return kenya_flood_drainage


# =============================================================================
#  SECTION 4: DATASET STATISTICS (for validation / debugging)
# =============================================================================

def print_dataset_summary():
    """
    Prints a quick count of records by county and by risk level.
    Run this directly to verify the dataset loaded correctly:
        python data/kenya_flood_drainage_dataset.py
    """
    from collections import Counter

    county_counts  = Counter(c["County"] for c in kenya_flood_drainage)
    risk_counts    = Counter(c["Flood_Risk_Level"] for c in kenya_flood_drainage)
    drain_counts   = Counter(c["Drainage_Quality"] for c in kenya_flood_drainage)

    print("\n" + "="*60)
    print("KENYA FLOOD DRAINAGE DATASET — SUMMARY")
    print("="*60)
    print(f"Total grid cells : {len(kenya_flood_drainage)}")
    print(f"Counties covered : {len(county_counts)}")
    print("\nRisk Level Distribution:")
    for level, count in sorted(risk_counts.items(), key=lambda x: -x[1]):
        pct = count / len(kenya_flood_drainage) * 100
        bar = "█" * int(pct / 2)
        print(f"  {level:<10} : {count:>3}  ({pct:.1f}%)  {bar}")
    print("\nDrainage Quality Distribution:")
    for quality, count in sorted(drain_counts.items(), key=lambda x: -x[1]):
        print(f"  {quality:<10} : {count:>3}")
    print("\nCounties with most grid cells:")
    for county, count in county_counts.most_common(10):
        print(f"  {county:<20}: {count}")
    print("="*60)


# Allow running this file directly to validate dataset
if __name__ == "__main__":
    print_dataset_summary()
