import argparse
import datetime

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import DatasetDict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding

import wandb

WARMUP_STEPS = 5000
EPOCHS = 30
SAMPLE_SIZE = 200_000
BATCH_SIZE = 32
LR = 2e-4

# esm_model_name = (
#     "facebook/esm2_t33_650M_UR50D"  # Replace with the correct ESM2 model name
# )
# chem_model_name = (
#     "seyonec/ChemBERTa-zinc-base-v1"  # Replace with the correct ChemLLM model name
# )

esm_model_name = "/home/share/huadjyin/home/nikolamilicevic/.cache/huggingface/hub/models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c/"
chem_model_name = "/home/share/huadjyin/home/nikolamilicevic/.cache/huggingface/hub/models--seyonec--ChemBERTa-zinc-base-v1/snapshots/761d6a18cf99db371e0b43baf3e2d21b3e865a20/"


class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1, ffn_hidden_dim=2048):
        super(CrossAttentionLayer, self).__init__()
        self.protein_to_ligand_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ligand_to_protein_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        ffn_hidden_dim = embed_dim * 3
        self.ffn_protein = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.ReLU(),  # Non-linear activation
            nn.Linear(ffn_hidden_dim, embed_dim),
        )
        self.ffn_ligand = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.ReLU(),  # Non-linear activation
            nn.Linear(ffn_hidden_dim, embed_dim),
        )
        self.protein_norm = nn.LayerNorm(embed_dim)
        self.ligand_norm = nn.LayerNorm(embed_dim)
        self.ffn_protein_norm = nn.LayerNorm(embed_dim)
        self.ffn_ligand_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        protein_embedding,
        ligand_embedding,
        key_pad_mask_prot,
        key_pad_mask_ligand,
    ):
        # Protein attending to ligand
        attended_protein, _ = self.protein_to_ligand_attention(
            query=protein_embedding,
            key=ligand_embedding,
            value=ligand_embedding,
            key_padding_mask=key_pad_mask_ligand,
        )
        attended_protein = self.protein_norm(
            protein_embedding + attended_protein
        )  # Residual connection
        x_prot = self.ffn_protein(attended_protein)
        x_prot = self.ffn_protein_norm(attended_protein + self.dropout(x_prot))

        # Ligand attending to protein
        attended_ligand, _ = self.ligand_to_protein_attention(
            query=ligand_embedding,
            key=protein_embedding,
            value=protein_embedding,
            key_padding_mask=key_pad_mask_prot,
        )
        attended_ligand = self.ligand_norm(
            ligand_embedding + attended_ligand
        )  # Residual connection
        x_ligand = self.ffn_ligand(attended_ligand)
        x_ligand = self.ffn_ligand_norm(attended_ligand + self.dropout(x_ligand))
        return x_prot, x_ligand


class BindingAffinityModelWithMultiHeadCrossAttention(nn.Module):
    def __init__(self, esm_model_name, chem_model_name, num_layers=3, hidden_dim=1024):
        super().__init__()
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained ESM2 model for proteins
        self.esm_model = AutoModel.from_pretrained(esm_model_name)
        self.esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
        self.esm_model.eval()

        # Load pretrained ChemLLM for SMILES (ligands)
        self.ligand_model = AutoModel.from_pretrained(chem_model_name)
        self.ligand_tokenizer = AutoTokenizer.from_pretrained(chem_model_name)
        self.ligand_model.eval()

        # Disable gradient computation for both base models
        for param in self.esm_model.parameters():
            param.requires_grad = False

        for param in self.ligand_model.parameters():
            param.requires_grad = False

        # Protein and SMILES embedding dimensions
        self.protein_embedding_dim = self.esm_model.config.hidden_size
        self.ligand_embedding_dim = self.ligand_model.config.hidden_size

        self.project_to_common = nn.Linear(
            self.protein_embedding_dim, self.ligand_embedding_dim
        )

        self.layers = nn.ModuleList(
            [
                CrossAttentionLayer(embed_dim=self.ligand_embedding_dim)
                for _ in range(num_layers)
            ]
        )

        self.ffn_ic50 = nn.Sequential(
            nn.Linear(2 * self.ligand_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        ligand_input_ids,
        ligand_attention_mask,
        protein_input_ids,
        protein_attention_mask,
    ):
        # Protein embedding
        # protein_inputs = self.esm_tokenizer(protein_sequence, return_tensors="pt")
        protein_inputs = {
            "input_ids": protein_input_ids,
            "attention_mask": protein_attention_mask,
        }
        # perform in FP16 for lower memory usage (matmuls)
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                protein_outputs = self.esm_model(**protein_inputs)
        special_tokens_mask_prot = (
            (protein_inputs["input_ids"] == self.esm_tokenizer.cls_token_id)
            | (protein_inputs["input_ids"] == self.esm_tokenizer.eos_token_id)
            | (protein_inputs["input_ids"] == self.esm_tokenizer.pad_token_id)
        )
        protein_embedding = protein_outputs.last_hidden_state

        # SMILES embedding
        ligand_inputs = {
            "input_ids": ligand_input_ids,
            "attention_mask": ligand_attention_mask,
        }
        special_tokens_mask_ligand = (
            (ligand_inputs["input_ids"] == self.ligand_tokenizer.bos_token_id)
            | (ligand_inputs["input_ids"] == self.ligand_tokenizer.eos_token_id)
            | (ligand_inputs["input_ids"] == self.ligand_tokenizer.pad_token_id)
        )

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                ligand_outputs = self.ligand_model(**ligand_inputs)
        ligand_embedding = ligand_outputs.last_hidden_state

        # project embeddings to same dimension
        protein_embedding = self.project_to_common(protein_embedding)

        for layer in self.layers:
            protein_embedding, ligand_embedding = layer(
                protein_embedding,
                ligand_embedding,
                special_tokens_mask_prot,
                special_tokens_mask_ligand,
            )

        # Perform mean pooling
        ligand_embedding = (
            ligand_embedding * ~special_tokens_mask_ligand.unsqueeze(dim=-1)
        ).mean(dim=1)
        protein_embedding = (
            protein_embedding * ~special_tokens_mask_prot.unsqueeze(dim=-1)
        ).mean(dim=1)
        # Combine embeddings
        combined = torch.cat([protein_embedding, ligand_embedding], dim=1)
        ic50_prediction = self.ffn_ic50(combined)
        return ic50_prediction


class CustomDataCollator:
    def __init__(self, chem_collator, esm_collator):
        self.chem_collator = chem_collator
        self.esm_collator = esm_collator

    def __call__(self, batch):
        batch_ligand = [
            {
                "input_ids": b["ligand_input_ids"],
                "attention_mask": b["ligand_attention_mask"],
            }
            for b in batch
        ]
        batch_protein = [
            {
                "input_ids": b["protein_input_ids"],
                "attention_mask": b["protein_attention_mask"],
            }
            for b in batch
        ]

        collated_chem = self.chem_collator(batch_ligand)
        collated_esm = self.esm_collator(batch_protein)

        return {
            "ligand_input_ids": collated_chem["input_ids"],
            "ligand_attention_mask": collated_chem["attention_mask"],
            "protein_input_ids": collated_esm["input_ids"],
            "protein_attention_mask": collated_esm["attention_mask"],
            # "ic50": torch.tensor([x["ic50_scaled"] for x in batch]),
            "ic50": torch.tensor([x["ic50"] for x in batch]),
        }


def main(timestamp, args):
    wandb.login()
    wandb.init(
        project="affinity",
        name=args.wandb_name,
        config={
            "batch_size": 8,
            "dataset": "testing_100k",
        },
    )

    chem_tokenizer = AutoTokenizer.from_pretrained(chem_model_name)
    esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_name)

    if (
        "binding" in args.ds_path
    ):  # for testing, assuming name such as binding_200k etc.
        df = pd.read_csv("/home/share/huadjyin/home/nikolamilicevic/BindingDB_fin.csv")
        df = df.sample(n=SAMPLE_SIZE, random_state=42)
        df = df.rename(
            columns={
                "Ligand SMILES": "ligand",
                "BindingDB Target Chain Sequence": "protein",
                "IC50 (nM)": "ic50",
            }
        )
        ic50 = df["ic50"][~df["ic50"].isna()]
        less_than = ic50.str.contains("<")
        greater_than = ic50.str.contains(">")
        ic50 = ic50[~(less_than | greater_than)]
        ic50n = ic50.astype(float)
        df = df.loc[ic50n.index]
        #  IC50 values are logarithmic in nature;
        # a compound with an IC50 of 10 nM is 10x more potent than one with 100 nM
        df["ic50"] = ic50n
        df = df[df["ic50"] != 0.0]
        df.loc[:, "ic50"] = np.log(df["ic50"])

        print(f"Ligands total: {len(df['ligand'].values)}")
        print(f"Ligands unique {len(set(df['ligand'].values))}")
        print(f"Proteins total {len(df['protein'].values)}")
        print(f"Proteins unique {len(set(df['protein'].values))}")

        mean_ic50 = np.mean(df["ic50"])
        y_pred_dummy = np.full_like(df["ic50"], mean_ic50)
        loss = nn.MSELoss()
        dummy_loss = loss(torch.tensor(y_pred_dummy), torch.tensor(df["ic50"].values))
        print(f"Dummy model MSE predicting mean ic50: {dummy_loss}")

        sns.histplot(df["ic50"], bins=100, color="blue")
        plt.savefig("histplot.png", dpi=300, bbox_inches="tight")

        dataset = datasets.Dataset.from_pandas(df.reset_index(drop=True))

        print(f"Maximum ligand length {df['ligand'].str.len().max()}")
        print(f"Minimum ligand length {df['ligand'].str.len().min()}")
        print(f"Mean ligand length {df['ligand'].str.len().mean():.2f}")
        print(f"Median ligand length {df['ligand'].str.len().median()}")
        print(f"Maximum protein length {df['protein'].str.len().max()}")
        print(f"Minimum protein length {df['protein'].str.len().min()}")
        print(f"Mean protein length {df['protein'].str.len().mean():.2f}")
        print(f"Median protein length {df['protein'].str.len().median()}")

        # Data split
        # Filter protein sequences longer than 1024 and ligands longer than 512
        dataset = dataset.filter(lambda x: len(x["protein"]) < 1024)
        dataset = dataset.filter(lambda x: len(x["ligand"]) < 512)

        dataset_train_test = dataset.train_test_split(test_size=0.2)
        dataset_test_val = dataset_train_test["test"].train_test_split(test_size=0.5)
        dataset_dict = {
            "train": dataset_train_test["train"],
            "test": dataset_test_val["train"],
            "validation": dataset_test_val["test"],
        }
        dataset = DatasetDict(dataset_dict)

        # Tokenization
        # Tokenization of ligands and protein sequences
        print(f"chem tokenizer is fast: {chem_tokenizer.is_fast}")
        print(f"esm tokenizer is fast: {esm_tokenizer.is_fast}")

        def tokenize_ligands(examples):
            toks = chem_tokenizer(examples["ligand"], truncation=True)
            return {
                "ligand_input_ids": toks["input_ids"],
                "ligand_attention_mask": toks["attention_mask"],
            }

        def tokenize_proteins(examples):
            toks = esm_tokenizer(examples["protein"], truncation=True)
            return {
                "protein_input_ids": toks["input_ids"],
                "protein_attention_mask": toks["attention_mask"],
            }

        tokenized_dataset = dataset.map(tokenize_proteins, batched=True)
        tokenized_dataset = tokenized_dataset.map(tokenize_ligands, batched=True)
        tokenized_dataset.save_to_disk("binding_ds_100k")
    else:
        tokenized_dataset = datasets.load_from_disk(args.ds_path)

    # # Z-Score normalization of ic50
    # scaler = StandardScaler()
    # ic50_train = np.array(tokenized_dataset["train"]["ic50"]).reshape(-1, 1)
    # ic50_test = np.array(tokenized_dataset["test"]["ic50"]).reshape(-1, 1)
    # ic50_validation = np.array(tokenized_dataset["validation"]["ic50"]).reshape(-1, 1)
    # ic50_train_scaled = scaler.fit_transform(ic50_train)
    # ic50_test_scaled = scaler.transform(ic50_test)
    # ic50_validation_scaled = scaler.transform(ic50_validation)

    # tokenized_dataset["train"] = tokenized_dataset["train"].add_column(
    #     "ic50_scaled", ic50_train_scaled.flatten()
    # )
    # tokenized_dataset["test"] = tokenized_dataset["test"].add_column(
    #     "ic50_scaled", ic50_test_scaled.flatten()
    # )
    # tokenized_dataset["validation"] = tokenized_dataset["validation"].add_column(
    #     "ic50_scaled", ic50_validation_scaled.flatten()
    # )

    # Custom data collator
    chem_collator = DataCollatorWithPadding(tokenizer=chem_tokenizer)
    esm_collator = DataCollatorWithPadding(tokenizer=esm_tokenizer)

    bs = BATCH_SIZE
    collator = CustomDataCollator(
        chem_collator=chem_collator, esm_collator=esm_collator
    )
    train_dataloader = DataLoader(
        tokenized_dataset["train"], batch_size=bs, collate_fn=collator
    )
    test_dataloader = DataLoader(
        tokenized_dataset["test"], batch_size=bs, collate_fn=collator
    )
    val_dataloader = DataLoader(
        tokenized_dataset["validation"], batch_size=bs, collate_fn=collator
    )

    # Training loop
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            # Linear warmup
            return step / WARMUP_STEPS
        else:
            remaining_steps = total_steps - WARMUP_STEPS
            decay_step = step - WARMUP_STEPS
            return max(0.5 * LR, 1.0 - 0.5 * (decay_step / remaining_steps))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    model = BindingAffinityModelWithMultiHeadCrossAttention(
        esm_model_name, chem_model_name
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    total_steps = EPOCHS * len(train_dataloader)

    def train_model(model, train_dataloader, val_dataloader):
        step = 0

        for epoch in range(EPOCHS):
            print(f"Epoch: {epoch + 1}/{EPOCHS}")
            model.train()
            train_loss = 0.0
            train_progress = tqdm(train_dataloader, desc="Training")

            for batch in train_progress:
                ligand_input_ids = batch["ligand_input_ids"].to(device)
                ligand_attention_mask = batch["ligand_attention_mask"].to(device)
                protein_input_ids = batch["protein_input_ids"].to(device)
                protein_attention_mask = batch["protein_attention_mask"].to(device)
                targets = batch["ic50"].unsqueeze(dim=-1).to(device)
                optimizer.zero_grad()
                preds = model(
                    ligand_input_ids,
                    ligand_attention_mask,
                    protein_input_ids,
                    protein_attention_mask,
                )
                loss = criterion(preds, targets)
                loss.backward()
                train_loss += loss.item()
                step += 1
                optimizer.step()
                scheduler.step()

                if step % 100 == 0:
                    wandb.log({"train_loss": loss.item()})
                    wandb.log({"lr": optimizer.param_groups[0]["lr"]})

            train_loss /= len(train_dataloader)
            print(f"Epoch: {epoch} Train loss: {train_loss}")

            if epoch > 20:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    f"affinity_{timestamp}_{epoch}.pt",
                )

            model.eval()
            val_loss = 0.0
            val_progress = tqdm(val_dataloader, desc="Validation")
            with torch.no_grad():
                for batch in val_progress:
                    ligand_input_ids = batch["ligand_input_ids"].to(device)
                    ligand_attention_mask = batch["ligand_attention_mask"].to(device)
                    protein_input_ids = batch["protein_input_ids"].to(device)
                    protein_attention_mask = batch["protein_attention_mask"].to(device)
                    targets = batch["ic50"].unsqueeze(dim=-1).to(device)
                    preds = model(
                        ligand_input_ids,
                        ligand_attention_mask,
                        protein_input_ids,
                        protein_attention_mask,
                    )
                    loss = criterion(preds, targets)
                    val_loss += loss.item()

            val_loss /= len(val_dataloader)
            wandb.log({"val_loss": val_loss})
            print(f"Epoch: {epoch} Val loss: {val_loss}")

    train_model(model, train_dataloader, val_dataloader)

    torch.save(model.state_dict(), f"affinity_{timestamp}.pt")

    def evaluate_model(model, test_loader):
        model.eval()
        all_predictions = []
        all_targets = []
        test_progress = tqdm(test_loader, desc="Test set")
        with torch.no_grad():
            for batch in test_progress:
                ligand_input_ids = batch["ligand_input_ids"].to(device)
                ligand_attention_mask = batch["ligand_attention_mask"].to(device)
                protein_input_ids = batch["protein_input_ids"].to(device)
                protein_attention_mask = batch["protein_attention_mask"].to(device)
                targets = batch["ic50"].unsqueeze(dim=-1).to(device)
                preds = model(
                    ligand_input_ids,
                    ligand_attention_mask,
                    protein_input_ids,
                    protein_attention_mask,
                )
                all_targets.append(targets)
                all_predictions.append(preds)

        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)

        loss = torch.nn.MSELoss()
        print(
            f"Test set mean squared error (MSE): {loss(all_predictions, all_targets)}"
        )

        mse = mean_squared_error(all_targets.cpu(), all_predictions.cpu())
        mae = mean_absolute_error(all_targets.cpu(), all_predictions.cpu())
        r2 = r2_score(all_targets.cpu(), all_predictions.cpu())
        print(f"MSE: {mse}, MAE: {mae}, R²: {r2}")

    evaluate_model(model, test_loader=test_dataloader)

    print("Finished script")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script that trains model on protein-ligand interactions represented via ic50. Make sure you have wandb on your machine and you are logged in."
    )
    parser.add_argument(
        "--wandb_name",
        help="Name of run on wandb platform",
        type=str,
        required=False,
        default="affinity_script",
    )

    parser.add_argument(
        "--ds_path",
        help="Path of the dataset in huggingface datasets format",
        type=str,
        required=True,
        default="binding_ds_300k",
    )

    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    print(timestamp)
    main(timestamp, args)
