import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from llm.layers.model import Model
from llm.utils.load_data import get_dataloaders
from torch import nn, optim, exp
import torch 


class LLM(L.LightningModule):
    def __init__(self, vocab_size, embed_dim, num_heads, tokenizer, pad_idx=0, max_length=1024, num_blocks=6):
        super().__init__()
        self.model = Model(vocab_size, embed_dim, num_heads, pad_idx, max_length, num_blocks)

        self.loss_fn = nn.CrossEntropyLoss()
        self.tokenizer = tokenizer

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-4)
    
    def forward(self, input_ids, padding_mask=None):
        if padding_mask is None:
            padding_mask = torch.ones_like(input_ids, device=self.device)
        return self.model(input_ids, padding_mask)
    
    def training_step(self, batch):
        input_ids, labels = batch['input_ids'], batch['labels']

        logits = self(input_ids)
        
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        perplexity = exp(loss)

        self.log_dict({
            "Train/Loss": loss,
            "Train/Perplexity": perplexity
        }, on_step=True, on_epoch=True, prog_bar=True)
        return loss 
    
    def validation_step(self, batch):
        input_ids, labels = batch['input_ids'], batch['labels']

        logits = self(input_ids)

        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        perplexity = exp(loss)

        self.log_dict({
            "Validation/Loss": loss,
            "Validation/Perplexity": perplexity
        }, prog_bar=True)
        return loss 



if __name__=="__main__":
    (train_dl, test_dl), tokenizer = get_dataloaders('./data', 8, 8)

    llm = LLM(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=512,
        num_heads=4,
        tokenizer=tokenizer,
        pad_idx=tokenizer.token_to_id("[PAD]"),
        max_length=1024,
        num_blocks=6
    )

    trainer = L.Trainer(
        max_epochs=1000,
        logger=WandbLogger('llm', project='LLM'),
        precision='16-mixed',
        callbacks=ModelCheckpoint("./checkpoints", monitor="Validation/Loss", mode="min", save_top_k=1)
    )
    trainer.fit(llm, train_dl, test_dl)