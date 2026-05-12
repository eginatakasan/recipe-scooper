LABEL2ID = {
    "O": 0,
    "TITLE": 1,
    "INGREDIENT": 2,
    "STEP": 3,
    "AUTHOR": 4,
    "DATE": 5,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}

MODEL_NAME = "microsoft/markuplm-base"
MAX_LENGTH = 512
