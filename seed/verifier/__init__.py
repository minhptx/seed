from .infotab import InfotabVerifier
from .tapas import TapasVerifier

class Verifier:
    def __init__(self) -> None:
        pass

    @staticmethod
    def get(verifier_model: str, *args, **kwargs):
        if verifier_model == "infotab":
            class_instance = InfotabVerifier
        elif verifier_model == "tapas":
            class_instance = TapasVerifier

        return class_instance(*args, **kwargs)

    