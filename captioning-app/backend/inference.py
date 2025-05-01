import torch
import pickle
from model import EncoderCNN
from model import DecoderRNN
from utils.image_utils import load_and_preprocess_image
from utils.video_utils import sample_video_frames, preprocess_video_frames

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("models/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
inv_vocab = {idx: word for word, idx in vocab.items()}

# Load model
encoder = EncoderCNN(embed_size=256).to(device)
decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=len(vocab)).to(device)
checkpoint = torch.load("models/unified_model_epoch_50.pth", map_location=device,weights_only=False)
encoder.load_state_dict(checkpoint["encoder_state_dict"])
decoder.load_state_dict(checkpoint["decoder_state_dict"])
encoder.eval()
decoder.eval()

def generate_caption(tensor_input, is_video=False):
    with torch.no_grad():
        features = encoder(tensor_input.to(device))

    caption = [vocab["<start>"]]
    for _ in range(20):
        input_tensor = torch.tensor(caption).unsqueeze(0).to(device)
        embeddings = decoder.embedding(input_tensor)
        h0 = decoder.init_h(features).unsqueeze(0)
        c0 = decoder.init_c(features).unsqueeze(0)
        hiddens, _ = decoder.lstm(embeddings, (h0, c0))
        outputs = decoder.linear(hiddens)
        predicted = outputs.argmax(2)[:, -1].item()
        caption.append(predicted)
        if predicted == vocab["<end>"]:
            break
    return " ".join([inv_vocab[idx] for idx in caption[1:-1]])

def generate_caption_image(image_path):
    tensor_input = load_and_preprocess_image(image_path)
    return generate_caption(tensor_input)

def generate_caption_video(video_path):
    frames = sample_video_frames(video_path)
    frames = preprocess_video_frames(frames)
    frames = frames.unsqueeze(0)  # (1, T, C, H, W)
    return generate_caption(frames, is_video=True)
