from pydantic import BaseModel


class CreditRiskRequest(BaseModel):

    TotalTransactionAmount: float

    AverageTransactionAmount: float

    TransactionCount: int

    StdTransactionAmount: float

    FraudResult: float