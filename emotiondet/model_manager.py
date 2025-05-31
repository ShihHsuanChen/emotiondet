import json
from typing import Dict, Tuple, List


def check_requirements():
    import torch
    import numpy
    import transformers
    import tiktoken
    import google.protobuf
    from .model import create_model


class ModelManager:
    def __init__(self, model_path: str, max_length: int = 80, batch_size: int = 1, device: str = 'cpu'):
        self.model = None
        self.tokenizer = None
        
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device

    def _load_train_config(self, train_config_path: str):
        with open(train_config_path, 'r') as fp:
            return json.load(fp)

    def load_model(self):
        import torch
        from .model import create_model # Lazy
        
        self.model, self.tokenizer = create_model(self.model_path)
        self.model.to(self.device).eval()

    def check_model_loaded(self):
        assert self.model is not None and self.tokenizer is not None, \
            'Model or tokenizer not loaded. Please call load_model() first.'

    def _preprocess(self, input_text: str) -> str:
        return input_text.lower()
        
    def single_inference(self, text: str) -> Tuple[str, Dict[str, float]]:
        import torch
        import torch.nn.functional as F
        import numpy as np
        
        self.check_model_loaded()

        id2label = self.model.config.id2label
        token = self.tokenizer(self._preprocess(text), truncation=True, padding="max_length", max_length=self.max_length)
        token = {k: torch.tensor([v], device=self.device) for k, v in token.items()}

        with torch.no_grad():
            outputs = self.model(**token)
            y_prob = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            y_pred = np.argmax(y_prob)
        prob = {v: float(y_prob[k]) for k, v in id2label.items()}
        pred = id2label[y_pred]
        return pred, prob

    def batch_inference(self, text_list: List[str]) -> List[Tuple[str, Dict[str, float]]]:
        import torch
        import torch.nn.functional as F
        import numpy as np
        
        self.check_model_loaded()

        id2label = self.model.config.id2label

        if len(text_list) == 0:
            return []
            
        res = []
        for i in range(0, len(text_list), self.batch_size):
            tokens = [
                self.tokenizer(self._preprocess(text), truncation=True, padding="max_length", max_length=self.max_length)
                for text in text_list[i:min(i+self.batch_size, len(text_list))]
            ]
            batch = {
                k: torch.tensor([token[k] for token in tokens], device=self.device)
                for k in tokens[0].keys()
            }
            with torch.no_grad():
                outputs = self.model(**batch)
                y_prob = F.softmax(outputs.logits, dim=1).cpu().numpy()
                y_pred = np.argmax(y_prob, axis=1)
            res.extend([
                (id2label.get(_y_pred), {v: float(_y_prob[k]) for k, v in id2label.items()})
                for _y_prob, _y_pred in zip(y_prob, y_pred)
            ])
        return res
