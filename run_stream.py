import torch
import torch.nn.functional as F
from InternVideo2.multi_modality.models.backbones.internvideo2.internvideo2_clip_recycle import StreamingInternVideo2Student
from InternVideo2.multi_modality.models.backbones.internvideo2.video_mamba_block import CrossMambaFiLM


def dummy_text_encode(prompt: str, dim: int):
    # Placeholder text encoder returning random features
    torch.manual_seed(len(prompt))
    return torch.randn(1, dim)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    student_config = {
        "vit_lite_model_name": "vit_b16",
        "vit_lite_proj_dim": 512,
        "vit_lite_embed_dim": 768,
        "rnn_type": 'cross_mamba_film',
        "rnn_hidden_size": 512,
        "rnn_num_layers": 1,
        "rnn_dropout": 0.0,
        "fc_hidden_layers": [256],
        "teacher_clip_embed_dim": 768,
    }

    model = StreamingInternVideo2Student(**student_config).to(device)
    model.eval()

    # Prepare FiLM params from text prompt
    prompt_vec = dummy_text_encode("a person riding a horse", student_config["teacher_clip_embed_dim"]).to(device)
    gamma, beta = model.rnn.prepare_prompt(prompt_vec)

    hidden = model.init_hidden(batch_size=1, device=device)

    for step in range(3):
        frame = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            out, hidden = model(frame, hidden, gamma, beta)
        score = F.cosine_similarity(out, prompt_vec, dim=-1)
        print(f"step {step} score {score.item():.3f}")


if __name__ == "__main__":
    main()
