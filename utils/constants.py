"""Define constant values used thoughout the project."""

PADDING_TOKEN = "<blank>"
START_OF_SENTENCE_TOKEN = "<s>"
END_OF_SENTENCE_TOKEN = "</s>"
UNKNOWN_TOKEN = "<unk>"

PADDING_ID = 0
START_OF_SENTENCE_ID = 1
END_OF_SENTENCE_ID = 2
NUM_OOV_BUCKETS = 1  # The number of additional unknown tokens <unk>.

INPUT_IDS = "input_ids"
INPUT_LENGTH = "input_length"
LABEL_IDS_IN = "label_ids_in"
LABEL_IDS_OUT = "label_ids_out"
LENGTH = "sequence_length"
LABEL_OUT = "y"

LM_VAR_SCOPE = "LanguageModel"
S2S_VAR_SCOPE = "Seq2SentiSeq"
CLS_VAR_SCOPE = "Classifier"
REG_VAR_SCOPE = "Regressor"

REWARD = "reward"

# names of decode type
RANDOM = "random"
GREEDY = "greedy"
BEAM = "beam"
MC_SEARCH = "MC_search"

# Standard names for model modes (make sure same to tf.estimator.ModeKeys.TRAIN).

TRAIN = 'train'
DUAL_TRAIN = 'dual_train'
EVAL = 'eval'
INFER = 'infer'
TEST = 'test'

MIN_SENT = 1
MAX_SENT = 5
SENT_LIST = [1, 2, 3, 4, 5]

