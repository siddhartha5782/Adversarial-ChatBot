import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import json
import argparse
from tqdm import tqdm

class QuoteDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=64):
        with open(path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Format: User: [Prompt] \n Bot: [Response] <|endoftext|>
        text = f"User: {item['prompt']}\nBot: {item['response']}<|endoftext|>"
        enc = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        return {k: v.squeeze() for k, v in enc.items()}

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)
    model.train()

    dataset = QuoteDataset(args.data_path, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    print(f"Training {args.mode} model on {args.data_path}...")

    for epoch in range(args.epochs):
        loop = tqdm(loader, leave=True)
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids.clone() # Standard language modeling objective

            if args.mode == 'standard':
                # --- TRADITIONAL METHOD ---
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                model.zero_grad()

            elif args.mode == 'freelb':
                # --- PROPOSED METHOD: FreeLB ---
                # 1. Get Embeddings
                embeddings = model.transformer.wte(input_ids)
                
                # 2. Initialize Delta (Perturbation)
                delta = torch.zeros_like(embeddings).uniform_(-args.adv_init_mag, args.adv_init_mag)
                delta.requires_grad = True
                
                # 3. Adversarial Loop
                for a_step in range(args.adv_steps):
                    # Inject perturbation into embeddings
                    # We have to bypass the standard model.forward() embedding lookup
                    # Ideally, we hook into the model, but for DistilGPT2 hack:
                    # We add delta to inputs_embeds if passing embeddings directly
                    
                    outputs = model(inputs_embeds=embeddings + delta, 
                                  attention_mask=attention_mask, 
                                  labels=labels)
                    loss = outputs.loss / args.adv_steps 
                    # loss.backward() # Accumulate gradients
                    loss.backward(retain_graph=(a_step < args.adv_steps - 1))
                    if a_step < args.adv_steps - 1:
                        # Update Delta (Ascent)
                        delta_grad = delta.grad.detach()
                        delta.data = delta.data + args.adv_lr * delta_grad
                        # Projection (Clip delta to keep it small)
                        delta.data = torch.clamp(delta.data, -args.adv_max_norm, args.adv_max_norm)
                        delta.grad.zero_()
                
                # 4. Update Model Parameters (Descent)
                optimizer.step()
                model.zero_grad()

            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())

    # Save
    model.save_pretrained(f"/content/drive/MyDrive/anc/gemini/model_{args.mode}")
    tokenizer.save_pretrained(f"/content/drive/MyDrive/anc/gemini/model_{args.mode}")
    model.save_pretrained(f"./model_{args.mode}")
    tokenizer.save_pretrained(f"./model_{args.mode}")
    print(f"Saved model_{args.mode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=['standard', 'freelb'])
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    # FreeLB Hyperparams
    parser.add_argument("--adv_steps", type=int, default=3)
    parser.add_argument("--adv_lr", type=float, default=1e-1)
    parser.add_argument("--adv_init_mag", type=float, default=0.05)
    parser.add_argument("--adv_max_norm", type=float, default=0.2)
    
    args = parser.parse_args()
    train(args)