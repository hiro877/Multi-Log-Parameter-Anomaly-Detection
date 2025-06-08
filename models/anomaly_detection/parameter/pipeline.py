import json
import torch
from tokenizers import BertWordPieceTokenizer
from transformers import BertForMaskedLM, AdamW, BertConfig
from torch.utils.data import DataLoader

from .model_utils import (
    MaskedTextDataset,
    MaskedTextTestDataset,
    DataHandler,
    ModelTrainer,
    ModelTester,
)


class ParameterADPipeline:
    """Pipeline for training and evaluating the parameter anomaly detection model."""

    def __init__(self, config_path: str):
        """Load configuration from ``config_path``."""
        with open(config_path, "r") as f:
            self.params = json.load(f)

        tokenizer_file = (
            f"models/anomaly_detection/parameter/trained_tokenizer/{self.params['tokenizer_dir']}/vocab.txt"
        )
        self.tokenizer = BertWordPieceTokenizer(tokenizer_file)

        if self.params.get("use_proposed_method", False):
            vocab_size = 200000
        else:
            vocab_size = self.tokenizer.get_vocab_size()

        self.model_config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=512,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertForMaskedLM(self.model_config).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.params["learning_rate"])

    def train(self):
        """Train the model using ``ModelTrainer``."""
        train_path = (
            f"datasets_for_models/sample_param/train/{self.params['train_data_path']}.txt"
        )
        train_dataset = MaskedTextDataset(
            train_path,
            self.tokenizer,
            self.params.get("use_proposed_method", False),
            self.params.get("learn_positinal_info", False),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.params["batch_size"],
            shuffle=True,
            collate_fn=DataHandler.collate_batch,
        )

        trainer = ModelTrainer(
            self.model,
            train_loader,
            self.optimizer,
            self.device,
            self.params["saved_model_dir"],
            self.params["epochs"],
        )
        if self.params.get("load_model_path"):
            trainer.load_model(self.params["load_model_path"], self.model_config)
        trainer.train()

    def evaluate(self):
        """Evaluate the model using ``ModelTester``."""
        test_path = (
            f"datasets_for_models/sample_param/test/{self.params['test_data_dir']}"  # noqa: E501
            f"dataset_test_{self.params['param_state'].lower()}.txt"
        )
        test_dataset = MaskedTextTestDataset(test_path, self.tokenizer)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.params["batch_size"],
            shuffle=False,
        )
        tester = ModelTester(
            self.model,
            self.tokenizer,
            test_loader,
            self.params.get("use_proposed_method", False),
            self.device,
        )
        results = tester.test(
            self.params["param_state"],
            self.params["thre_AD"],
            "results/miss_result.txt",
        )
        print("Evaluation Results:", results)
        return results
