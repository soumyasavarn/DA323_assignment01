import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --------------------------
# 1) MODEL DEFINITION
# --------------------------
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=1):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden
    
    def init_hidden(self, batch_size, device="cpu"):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

# --------------------------
# 2) HELPER FUNCTIONS
# --------------------------
def one_hot_encode(sequence, vocab_size):
    """ Convert a sequence of character indices into one-hot vectors. """
    batch_size, seq_len = sequence.shape
    out = torch.zeros(batch_size, seq_len, vocab_size)
    for b in range(batch_size):
        for s in range(seq_len):
            out[b, s, sequence[b, s]] = 1.0
    return out

def get_batches(arr, batch_size, seq_length):
    """ Create a generator that returns batches of data (features, targets). """
    n_chars_per_batch = batch_size * seq_length
    n_batches = len(arr) // n_chars_per_batch
    arr = arr[:n_batches * n_chars_per_batch]
    arr = arr.reshape((batch_size, -1))
    
    for i in range(0, arr.shape[1] - seq_length, seq_length):
        x = arr[:, i:i+seq_length]
        y = arr[:, i+1:i+seq_length+1]
        yield x, y

def generate_text(model, start_str, char2idx, idx2char, n_chars=200, device="cpu"):
    """ Generate text from a trained model, given a start string. """
    model.eval()
    hidden = model.init_hidden(batch_size=1, device=device)
    
    # Convert start_str to character indices
    chars = [c for c in start_str]
    # Filter out any characters not in the vocabulary
    input_seq = torch.tensor([[char2idx[ch]] for ch in chars if ch in char2idx]).to(device)
    
    # Warm up the hidden state with the prompt
    for i in range(len(input_seq)):
        one_hot = one_hot_encode(input_seq[i].unsqueeze(0), len(char2idx)).to(device)
        _, hidden = model(one_hot, hidden)
    
    # The last character in the prompt
    out_char = chars[-1] if chars else ' '  # Fallback if prompt was empty
    
    # Generate new characters
    for _ in range(n_chars):
        if out_char not in char2idx:
            # If out_char not in vocab, break early
            break
        inp = torch.tensor([[char2idx[out_char]]]).to(device)
        one_hot_inp = one_hot_encode(inp, len(char2idx)).to(device)
        
        out, hidden = model(one_hot_inp, hidden)
        probs = nn.functional.softmax(out[0][-1], dim=0).detach().cpu().numpy()
        
        # Sample from the distribution
        char_idx = np.random.choice(len(probs), p=probs)
        out_char = idx2char[char_idx]
        chars.append(out_char)
    
    return "".join(chars)

# --------------------------
# 3) MAIN SCRIPT
# --------------------------
def main():
    # --------------------------
    # A) GET USER INPUT
    # --------------------------
    category = input("Enter category name (must match one of the .txt files in LexiVerse/): ")
    if not category.strip():
        print("No category provided. Exiting.")
        return

    prompt = input("Enter prompt (default: 'Hello'): ")
    if not prompt.strip():
        prompt = "Hello"
    
    try:
        epochs = int(input("Enter number of training epochs (default: 5): "))
    except ValueError:
        epochs = 5
    
    try:
        seq_length = int(input("Enter sequence length for training (default: 100): "))
    except ValueError:
        seq_length = 100
    
    try:
        generate_length = int(input("Enter number of characters to generate (default: 200): "))
    except ValueError:
        generate_length = 200

    # --------------------------
    # B) SETUP DEVICE
    # --------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")
    
    # --------------------------
    # C) LOAD DATA
    # --------------------------
    data_path = os.path.join("LexiVerse", f"{category}.txt")
    if not os.path.exists(data_path):
        print(f"Category file {data_path} does not exist. Exiting.")
        return
    
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Create character mappings
    chars = sorted(list(set(text)))
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)
    
    print(f"Loaded category '{category}' with {len(text)} characters.")
    print(f"Vocab size = {vocab_size}")
    
    # Convert entire text to indices
    encoded = np.array([char2idx[ch] for ch in text if ch in char2idx])
    
    # --------------------------
    # D) TRAINING SETUP
    # --------------------------
    batch_size = 64
    hidden_size = 256
    num_layers = 2
    
    model = CharLSTM(vocab_size, hidden_size, num_layers=num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # --------------------------
    # E) TRAIN THE MODEL
    # --------------------------
    model.train()
    print("\nStarting training...\n")
    for epoch in range(epochs):
        hidden = model.init_hidden(batch_size, device=device)
        epoch_loss = 0.0
        batch_count = 0
        
        for x, y in get_batches(encoded, batch_size, seq_length):
            batch_count += 1
            
            x = torch.tensor(x, dtype=torch.long).to(device)
            y = torch.tensor(y, dtype=torch.long).to(device)
            
            # One-hot encode the inputs
            x_onehot = one_hot_encode(x, vocab_size).to(device)
            
            optimizer.zero_grad()
            
            # Detach hidden state from previous iteration
            hidden = tuple([h.detach() for h in hidden])
            
            outputs, hidden = model(x_onehot, hidden)
            # Reshape outputs to (batch_size*seq_length, vocab_size)
            outputs = outputs.reshape(batch_size * seq_length, vocab_size)
            y = y.contiguous().view(-1)
            
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / batch_count if batch_count else 0
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # --------------------------
    # F) GENERATE TEXT
    # --------------------------
    print("\n--- Text Generation ---")
    model.eval()
    generated_text = generate_text(
        model, 
        start_str=prompt, 
        char2idx=char2idx, 
        idx2char=idx2char, 
        n_chars=generate_length, 
        device=device
    )
    
    print(f"\nPrompt: {prompt}")
    print(f"Generated Text:\n{generated_text}")

if __name__ == "__main__":
    main()
