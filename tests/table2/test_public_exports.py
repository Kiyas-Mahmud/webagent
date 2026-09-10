from web_agent.eval.table2 import SealedVerifier, SealedVerifierSink
from web_agent.eval.table2.sealed_verifier import SealedVerifier as ModuleVerifier


def test_sealed_verifier_compatibility_alias_is_public() -> None:
    assert SealedVerifier is SealedVerifierSink
    assert ModuleVerifier is SealedVerifierSink
