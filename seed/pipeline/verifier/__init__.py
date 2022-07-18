from .models.bart import BartVerifier
from .models.tapex import TapexVerifier
from .models.infotab import InfotabVerifier
from .models.tapas import TapasVerifier


class Verifier:
    name2class = {
        "infotab": InfotabVerifier,
        "tapas": TapasVerifier,
        "tapex": TapexVerifier,
        "bart": BartVerifier
    }

    def __init__(self) -> None:
        pass



    @staticmethod
    def get(verifier_model: str, *args, **kwargs):
        return Verifier.name2class[verifier_model](*args, **kwargs)

    