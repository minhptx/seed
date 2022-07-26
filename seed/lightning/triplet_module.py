from collections import defaultdict
import torch
from pytorch_lightning import (
    LightningModule,
)
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from yaml import load
import torchmetrics
import torch


class TripletPLTransformer(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        margin: float = 0.5,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(
            model_name_or_path, num_labels=num_labels
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, config=self.config, ignore_mismatched_sizes=True
        )

        self.metrics = defaultdict(dict)

        self.train_acc = torchmetrics.Accuracy(num_labels=num_labels)

        self.metrics["train"] = {"acc": self.train_acc}

        self.val_acc = torchmetrics.Accuracy(num_labels=num_labels)

        self.metrics["val"] = {"acc": self.val_acc}

    def forward(self, **inputs):
        return self.model(**{k: v.long() for k, v in inputs.items()})
        

    def training_step(self, batch, batch_idx):
        print("Batch", batch)
        outputs1 = self(**batch[0])
        outputs2 = self(**batch[1])
        loss = torch.softmax(outputs1, dim=1) - torch.softmax(outputs2, dim=2) + self.hparams.margin
        self.log("train_loss", loss, on_step=True, on_epoch=False)

        return {"loss": loss}

    def training_step_end(self, step_output):
        for name, metric in self.metrics["train"].items():
            self.log(
                f"train_{name}",
                metric(step_output["loss"] > 0, torch.tensor([1])),
                on_step=True,
                on_epoch=True,
            )
        return step_output['loss'].sum()


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        print("Batch", batch)
        outputs1 = self(**batch[0])
        outputs2 = self(**batch[1])
        loss = torch.softmax(outputs1, dim=1) - torch.softmax(outputs2, dim=2) + self.hparams.margin
        self.log("train_loss", loss, on_step=True, on_epoch=False)

        return {"loss": loss}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        test_loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]

        return {"loss": test_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        print("Batch", outputs)
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        for name, metric in self.metrics["val"].items():
            self.log(f"val_{name}", metric(outputs["loss"] > 0, torch.tensor([1])), prog_bar=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
