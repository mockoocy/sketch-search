from typing import Literal, Protocol

from pydantic import BaseModel, EmailStr
from sqlmodel import Session


class BeginRequest(BaseModel):
    email: EmailStr | None = None


class VerifyRequest(BaseModel):
    email: EmailStr
    code: str


class ChallengeOut(BaseModel):
    kind: Literal["challenge"] = "challenge"
    next_path: str = "/verify"


class IssueOut(BaseModel):
    kind: Literal["issue"] = "issue"
    access_token: str


type BeginOutcome = ChallengeOut | IssueOut


class AuthProvider(Protocol):
    name: str

    def begin(self, session: Session, req: BeginRequest) -> BeginOutcome: ...
    def verify(self, session: Session, req: VerifyRequest) -> IssueOut: ...
