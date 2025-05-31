
import torch
from transformers import DebertaV2Tokenizer, AutoTokenizer, DebertaV2ForSequenceClassification
from transformers.models.deberta_v2 import DebertaV2Tokenizer

LABELS = ['sadness','joy','love','anger','fear','surprise']
id2label = dict(enumerate(LABELS))
label2id = {v:k for k,v in id2label.items()}


def create_model(pretrained: str = "microsoft/deberta-v3-base"):
    # Initializing a model (with random weights) from the microsoft/deberta-v3-base style configuration
    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = DebertaV2ForSequenceClassification.from_pretrained(
        pretrained,
        label2id=label2id,
        id2label=id2label,
    )
    return model, tokenizer
