"""Shared experimental protocol for the ICT-4442 mini project.

Every model in this project MUST import from here. If two members segment
beats differently or use different splits, the comparison table is meaningless.
Nothing in this file may be changed by one member alone.
"""

# --- de Chazal et al. (2004) inter-patient division -------------------------
# The four paced records (102, 104, 107, 217) are excluded per AAMI EC57.
# No patient appears in both DS1 and DS2.
DS1 = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
       '122', '124', '201', '203', '205', '207', '208', '209', '215', '220',
       '223', '230']
DS2 = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202',
       '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
       '233', '234']

# Held out from DS1 *by patient* for validation, early stopping and the CNN's
# DE/PSO hyperparameter search. DS2 is touched exactly once, at the very end,
# by every model.
#
# These six records were chosen so the validation class proportions match DS1
# as a whole (S 1.87% vs 1.85%, V 7.41% vs 7.43%). Record 208 is deliberately
# NOT here: it contains 372 of DS1's 414 fusion beats, so moving it to
# validation would strip the F class out of training almost entirely.
DS1_VAL = ['108', '114', '116', '124', '207', '223']
DS1_TRAIN = [r for r in DS1 if r not in DS1_VAL]

# Consequence of that concentration: validation holds few F beats and no Q
# beats. Model selection (early stopping, DE/PSO) therefore optimises macro-F1
# over N/S/V/F. Final reporting still includes all five synopsis classes.
SEARCH_CLASSES = ['N', 'S', 'V', 'F']

# --- AAMI EC57 class mapping ------------------------------------------------
AAMI = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',   # normal / bundle branch
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',             # supraventricular ectopic
    'V': 'V', 'E': 'V',                                 # ventricular ectopic
    'F': 'F',                                           # fusion
    '/': 'Q', 'f': 'Q', 'Q': 'Q',                       # paced / unclassifiable
}

# The submitted synopsis commits to the five AAMI classes. Q is retained for
# reporting, but the report flags its very small support after excluding paced
# records, so conclusions are not overstated.
CLASSES = ['N', 'S', 'V', 'F', 'Q']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# --- Signal / segmentation --------------------------------------------------
FS = 360                 # Hz, native MIT-BIH sampling rate
LEAD = 'MLII'            # primary lead; record 114 has it in channel 1, not 0
PRE = 180                # samples before the annotated R peak (0.5 s)
POST = 180               # samples after                      (0.5 s)
WINDOW = PRE + POST      # 360 samples = 1.000 s per beat

# --- Signal cleaning (de Chazal et al., 2004) ------------------------------
# Baseline wander: two cascaded median filters (200 ms, then 600 ms) estimate
# the baseline, which is subtracted. Then a zero-phase low-pass at 35 Hz
# removes mains interference and muscle noise.
BASELINE_MED_MS = (200, 600)
LOWPASS_HZ = 35.0
LOWPASS_ORDER = 4

RANDOM_SEED = 42
# Every model is trained once per seed; reports give mean +/- std.
SEEDS = [42, 43, 44]

# --- Common evaluation protocol --------------------------------------------
# Accuracy alone is misleading here: predicting "N" for everything scores ~89%.
# Every model reports all of these on DS2.
METRICS = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1',
           'per_class_f1', 'confusion_matrix']
