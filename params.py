from typing import NamedTuple, Optional

class Params(NamedTuple):
    INPUT_DIM: Optional[int] = None 
    OUTPUT_DIM: Optional[int] = None
    HID_DIM: int = 200 # only works with KLD for small HID_DIM...
    DROPOUT: float = .1
    LEARNING_RATE: float = 1e-3
    FIXED_KAPPA: Optional[float] = 500
    TEACHER_FORCING_RATIO: float = 1.
    EMBEDDING_GRADIENT_NORM_CLIP: float = 0.1
    GRADIENT_NORM_CLIP: float = 5.
    NUM_DECODER_LAYERS: int = 4
    NUM_ENCODER_LAYERS: int = 2
    BIDIRECTIONAL_ENCODER: bool = False
    REVERSE_INPUT: bool = False
    MIXED_SRC_EMBEDDINGS: bool = False
    MIXED_TRG_EMBEDDINGS: bool = False
    FREEZE_LOADED_EMBEDDINGS: bool = True
    BPE_ENCODING_STEPS: int = 0 # if -1 searches for a conservative number of steps on its own
    ATTENTION_RETRIEVER: bool = False
    COPY_RETRIEVER: bool = False # will only be looked at if ATTENTION_RETRIEVER
    COPY_FORCING: bool = False # adds linear loss on `p_gen` to be small wherever copying is possible
    BPE_COPY_MASKING: bool = False # does not apply BPE to tokens that potentially could be pasted from source
    AUTO_BPE_MIN_FREQ: int = 1 # how often a token needs to appear in the training set to not be counted as unk in the validset test of auto bpe
    AST_BPE: bool = False # if BPE is used, should BPE be restricted to pairs of children of nodes in the SQL AST
    BPE_STEP_OVERRIDE_P_SCHEDULE: str = 'lambda epoch: 0.0'
    RNN_TYPE: str = 'LSTM'