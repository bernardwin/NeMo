from nemo.lightning.io.artifact import FileArtifact, DirOrStringArtifact
from nemo.lightning.io.mixin import track_io

__all__ = []
def extract_name(cls):
    return str(cls).split('.')[-1].rstrip('>').rstrip("'")

try:
    # Track HF tokenizers
    from transformers import AutoTokenizer as HfAutoTokenizer
    from transformers.models.llama.tokenization_llama import LlamaTokenizer
    from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
    for cls in [HfAutoTokenizer, LlamaTokenizer, LlamaTokenizerFast]:
        attr_names = ['vocab_file', 'merges_file', 'tokenizer_file', 'name_or_path']
        track_io(
            cls,
            artifacts=[
                FileArtifact(attr_name, required=False)
                for attr_name in attr_names
            ],
        )
        __all__.append(extract_name(cls))

    from nemo.collections.common.tokenizers import AutoTokenizer

    track_io(
        AutoTokenizer,
        artifacts=[
            FileArtifact("vocab_file", required=False),
            FileArtifact("merges_file", required=False),
            DirOrStringArtifact("pretrained_model_name", required=False),
        ],
    )
    __all__.append("AutoTokenizer")
except ImportError:
    pass


try:
    from nemo.collections.common.tokenizers import SentencePieceTokenizer

    track_io(SentencePieceTokenizer, artifacts=[FileArtifact("model_path")])
    __all__.append("SentencePieceTokenizer")
except ImportError:
    pass
