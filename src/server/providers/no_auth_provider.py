from sqlmodel import Session

from server.providers.auth_provider import (
    AuthProvider,
    BeginOutcome,
    BeginRequest,
    IssueOut,
    VerifyRequest,
)
from server.utils import make_token


class NoAuthProvider(AuthProvider):
    name = "none"

    def begin(
        self,
        session: Session,  # noqa: ARG002
        req: BeginRequest,
    ) -> BeginOutcome:
        subject = (req.email or "anonymous").lower()
        return IssueOut(access_token=make_token(subject))

    def verify(self, session: Session, req: VerifyRequest) -> IssueOut:
        raise NotImplementedError
