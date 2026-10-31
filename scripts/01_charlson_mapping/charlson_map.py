"""
Charlson Comorbidity Index category mapping for ICD-9-CM and ICD-10 codes,
implementing the Quan et al. (2005) "Coding algorithms for defining
comorbidities in ICD-9-CM and ICD-10 administrative data" enhanced ICD-9-CM
and ICD-10 coding algorithm (the standard, widely used mapping; the same
logic underlies the R `comorbidity` package's `charlson` mapping and the
Python `comorbidipy`/`icd` packages).

Input codes are expected WITHOUT the decimal point, matching MIMIC-IV's
`diagnoses_icd.csv` convention (e.g. "41401" not "414.01").

17 Charlson categories, in official weight order. Two "override" rules apply
when computing the composite score (handled in build_admission_features):
  - Diabetes with chronic complication supersedes diabetes without complication
  - Metastatic solid tumor supersedes malignancy
  - Moderate/severe liver disease supersedes mild liver disease
"""

import re

CHARLSON_WEIGHTS = {
    "Myocardial infarction": 1,
    "Congestive heart failure": 1,
    "Peripheral vascular disease": 1,
    "Cerebrovascular disease": 1,
    "Dementia": 1,
    "Chronic pulmonary disease": 1,
    "Rheumatic disease": 1,
    "Peptic ulcer disease": 1,
    "Mild liver disease": 1,
    "Diabetes without complication": 1,
    "Diabetes with complication": 2,
    "Hemiplegia or paraplegia": 2,
    "Renal disease": 2,
    "Malignancy": 2,
    "Moderate or severe liver disease": 3,
    "Metastatic solid tumor": 6,
    "AIDS/HIV": 6,
}

# Order matters for override logic
CATEGORY_ORDER = list(CHARLSON_WEIGHTS.keys())


def _num_range(lo, hi, ndigits=3):
    """Generate zero-padded numeric string prefixes lo..hi inclusive."""
    return [str(n).zfill(ndigits) for n in range(lo, hi + 1)]


# ---------------------------------------------------------------------------
# ICD-9-CM prefix rules (matched with startswith on the undotted code)
# ---------------------------------------------------------------------------
ICD9_RULES = {
    "Myocardial infarction": _num_range(410, 410) + ["412"],
    "Congestive heart failure": [
        "39891", "40201", "40211", "40291", "40401", "40403", "40411",
        "40413", "40491", "40493",
    ] + _num_range(4254, 4259, 4) + _num_range(428, 428),
    "Peripheral vascular disease": [
        "0930", "4373", "4471", "5571", "5579", "V434",
    ] + _num_range(440, 440) + _num_range(441, 441) + ["4439"],
    "Cerebrovascular disease": ["36234"] + _num_range(430, 438),
    "Dementia": ["2941", "3312"] + _num_range(290, 290),
    "Chronic pulmonary disease": _num_range(490, 505) + ["5064", "5081", "5088"],
    "Rheumatic disease": [
        "7100", "7101", "7104", "7140", "7141", "7142", "71481", "725",
    ],
    "Peptic ulcer disease": _num_range(531, 534),
    "Mild liver disease": [
        "5733", "5734", "5738", "5739", "V427",
    ] + _num_range(570, 571),
    "Diabetes without complication": _num_range(2500, 2503, 4) + ["2508", "2509"],
    "Diabetes with complication": _num_range(2504, 2507, 4),
    "Hemiplegia or paraplegia": [
        "3341", "3440", "3441", "3442", "3443", "3444", "3445", "3446", "3449",
    ] + _num_range(342, 343),
    "Renal disease": [
        "40301", "40311", "40391", "40402", "40403", "40412", "40413",
        "40492", "40493", "5880", "V420", "V451", "V56",
    ] + _num_range(582, 582) + _num_range(5830, 5837, 4) + _num_range(585, 586),
    "Malignancy": _num_range(140, 172) + _num_range(174, 195) + _num_range(200, 208) + ["2386"],
    "Moderate or severe liver disease": _num_range(4560, 4562, 4) + _num_range(5722, 5728, 4),
    "Metastatic solid tumor": _num_range(196, 199),
    "AIDS/HIV": _num_range(42, 44, 3),
}

# ---------------------------------------------------------------------------
# ICD-10-CM prefix rules (matched with startswith on the undotted code)
# ---------------------------------------------------------------------------
ICD10_RULES = {
    "Myocardial infarction": ["I21", "I22", "I252"],
    "Congestive heart failure": [
        "I099", "I110", "I130", "I132", "I255", "I420", "I425", "I426",
        "I427", "I428", "I429", "I43", "I50", "P290",
    ],
    "Peripheral vascular disease": [
        "I70", "I71", "I731", "I738", "I739", "I771", "I790", "I792",
        "K551", "K558", "K559", "Z958", "Z959",
    ],
    "Cerebrovascular disease": ["G45", "G46", "H340", "I60", "I61", "I62", "I63",
                                 "I64", "I65", "I66", "I67", "I68", "I69"],
    "Dementia": ["F00", "F01", "F02", "F03", "F051", "G30", "G311"],
    "Chronic pulmonary disease": ["J40", "J41", "J42", "J43", "J44", "J45", "J46",
                                   "J47", "J60", "J61", "J62", "J63", "J64", "J65",
                                   "J66", "J67", "J684", "J701", "J703"],
    "Rheumatic disease": ["M05", "M06", "M315", "M32", "M33", "M34", "M351", "M360"],
    "Peptic ulcer disease": ["K25", "K26", "K27", "K28"],
    "Mild liver disease": ["B18", "K700", "K701", "K702", "K703", "K709",
                            "K713", "K714", "K715", "K717", "K73", "K74",
                            "K760", "K762", "K763", "K764", "K768", "K769", "Z944"],
    "Diabetes without complication": ["E100", "E101", "E109", "E110", "E111", "E119",
                                        "E120", "E121", "E129", "E130", "E131", "E139",
                                        "E140", "E141", "E149"],
    "Diabetes with complication": ["E102", "E103", "E104", "E105", "E107",
                                     "E112", "E113", "E114", "E115", "E117",
                                     "E122", "E123", "E124", "E125", "E127",
                                     "E132", "E133", "E134", "E135", "E137",
                                     "E142", "E143", "E144", "E145", "E147"],
    "Hemiplegia or paraplegia": ["G041", "G114", "G801", "G802", "G81", "G82",
                                  "G830", "G831", "G832", "G833", "G834", "G839"],
    "Renal disease": ["I120", "I131", "N032", "N033", "N034", "N035", "N036", "N037",
                       "N052", "N053", "N054", "N055", "N056", "N057",
                       "N18", "N19", "N250", "Z490", "Z491", "Z492", "Z940", "Z992"],
    "Malignancy": ["C0", "C1", "C20", "C21", "C22", "C23", "C24", "C25", "C26",
                    "C30", "C31", "C32", "C33", "C34", "C37", "C38", "C39", "C40",
                    "C41", "C43", "C45", "C46", "C47", "C48", "C49", "C50", "C51",
                    "C52", "C53", "C54", "C55", "C56", "C57", "C58", "C60", "C61",
                    "C62", "C63", "C64", "C65", "C66", "C67", "C68", "C69", "C70",
                    "C71", "C72", "C73", "C74", "C75", "C76", "C81", "C82", "C83",
                    "C84", "C85", "C88", "C90", "C91", "C92", "C93", "C94", "C95", "C96", "C97"],
    "Moderate or severe liver disease": ["I850", "I859", "I864", "I982", "K704",
                                           "K711", "K721", "K729", "K765", "K766", "K767"],
    "Metastatic solid tumor": ["C77", "C78", "C79", "C80"],
    "AIDS/HIV": ["B20", "B21", "B22", "B24"],
}


def _compile_matcher(rules_dict):
    compiled = {}
    for cat, prefixes in rules_dict.items():
        compiled[cat] = tuple(sorted(set(prefixes)))
    return compiled


_ICD9_COMPILED = _compile_matcher(ICD9_RULES)
_ICD10_COMPILED = _compile_matcher(ICD10_RULES)


def clean_code(code):
    if code is None:
        return ""
    return re.sub(r"\.", "", str(code)).strip().upper()


def map_code_to_category(code, version):
    """Return the Charlson category name for a single ICD code, or None."""
    code = clean_code(code)
    if not code:
        return None
    rules = _ICD9_COMPILED if int(version) == 9 else _ICD10_COMPILED
    for cat, prefixes in rules.items():
        if code.startswith(prefixes):
            return cat
    return None
