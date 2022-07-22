from seed.lightning.module import PLTransformer


class PLTripletTransformer(PLTransformer):
    def forward(self, **inputs):
        return self.model(**{k: v.long() for k, v in inputs.items()})