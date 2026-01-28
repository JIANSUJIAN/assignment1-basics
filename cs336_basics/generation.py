import torch
import torch.nn.functional as F


def generate(model, input_ids, max_new_tokens, temperature=1.0, top_p=0.0, eos_token_id=None):
    model.eval()
    curr_ids = input_ids.clone()  # (batch, T)

    for _ in range(max_new_tokens):
        if hasattr(model, "max_seq_len") and curr_ids.shape[1] > model.max_seq_len:
            cond_ids = curr_ids[:, -model.max_seq_len]
        else:
            cond_ids = curr_ids

        with torch.no_grad():
            logits = model(cond_ids)
            next_token_logits = logits[:, -1, :]  # (batch, vocab)

        if temperature > 0:
            next_token_logits = next_token_logits / temperature
        else:
            # (Batch, 1)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            curr_ids = torch.cat((curr_ids, next_token), dim=1)
            continue

        # Top-p sampling
        if top_p > 0.0 and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits = next_token_logits.masked_fill(indices_to_remove, float("-inf"))

        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        curr_ids = torch.cat((curr_ids, next_token), dim=1)

        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

    return curr_ids
