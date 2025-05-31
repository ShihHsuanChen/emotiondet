from typing import Dict
from pydantic import BaseModel


class PredictResult(BaseModel):
    pred: str
    prob: Dict[str, float]
