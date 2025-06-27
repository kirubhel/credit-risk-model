from pydantic import BaseModel
from typing import Literal

class PredictionRequest(BaseModel):
    Amount: float
    Value: float
    ProductCategory: str
    ChannelId: str
    ProviderId: str
    PricingStrategy: int
    TransactionStartTime: str  # ISO format

class PredictionResponse(BaseModel):
    risk_probability: float
