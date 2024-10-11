import torch
import torchaudio
from huggingface_hub import hf_hub_download
from moshi.models import loaders
from safetensors.torch import load_file, save_file

def load_and_prepare_audio(file_path, target_length=24000*174):  # 2:54 = 174 seconds
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 24000:
        resampler = torchaudio.transforms.Resample(sample_rate, 24000)
        waveform = resampler(waveform)
    
    # Ensure mono audio
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Pad or trim to target length
    if waveform.shape[1] < target_length:
        waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.shape[1]))
    else:
        waveform = waveform[:, :target_length]
    
    return waveform.unsqueeze(0)  # Add batch dimension [B, C, T]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Mimi
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device=device)
    mimi.eval()  # Keep Mimi in eval mode as we're not fine-tuning it

    # Load Moshi
    moshi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
    moshi = loaders.get_moshi_lm(moshi_weight, device=device)
    state_dict = load_file(moshi_weight)
    moshi.load_state_dict(state_dict)
    moshi.train()

    # Load and prepare audio data
    audio = load_and_prepare_audio("path/to/your/audio.wav").to(device)

    # Encode audio using Mimi
    with torch.no_grad():
        encoded_audio = mimi.encode(audio)

    # Optimizer
    optimizer = torch.optim.AdamW(moshi.parameters(), lr=1e-5, weight_decay=0.01)

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass through Moshi
        output = moshi(encoded_audio)

        # Compute loss
        # Assuming Moshi outputs logits for the next token prediction
        loss = torch.nn.functional.cross_entropy(output.view(-1, output.size(-1)), encoded_audio.view(-1))

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    # Save the fine-tuned model
    save_file(moshi.state_dict(), "moshi_finetuned_audio_only.safetensors")
    print("Fine-tuned model saved as moshi_finetuned_audio_only.safetensors")

if __name__ == "__main__":
    main()