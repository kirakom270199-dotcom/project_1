import torch
from tqdm import tqdm
import evaluate
from transformers import AutoTokenizer


def tokens_to_text(tokens, tokenizer):
    """Конвертирует токены в текст"""
    if isinstance(tokens, torch.Tensor):
        toks = tokens.cpu().tolist()
    else:
        toks = list(tokens)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    toks = [t for t in toks if isinstance(t, int) and t >= 0 and t != -100 and t != pad_token_id]
    return tokenizer.decode(toks, skip_special_tokens=True, clean_up_tokenization_spaces=True)


@torch.no_grad()
def greedy_generate(model, prefix_ids, gen_len, device):
    """Жадная генерация текста"""
    model.eval()
    if gen_len <= 0:
        return []
    prefix_tensor = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated = prefix_tensor.clone()
    for _ in range(gen_len):
        logits = model(generated)
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_id], dim=1)
    return generated.squeeze(0).cpu().tolist()


@torch.no_grad()
def evaluate_lstm_generation(model, loader, tokenizer, device, max_examples=200, print_examples=3):
    """Валидация генерации с ROUGE метриками"""
    model.eval()
    rouge = evaluate.load("rouge")
    preds, refs = [], []
    printed = 0
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    for batch in tqdm(loader, desc="LSTM Evaluation"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch.get('attention_mask', None)
        labels = batch.get('labels', None)

        B, S = input_ids.size()

        if attention_mask is not None:
            real_lens = attention_mask.sum(dim=1).cpu().tolist()
        else:
            real_lens = [S] * B

        for i in range(B):
            if len(preds) >= max_examples:
                break

            real_len = int(real_lens[i])
            if real_len < 4:
                continue

            cut = max(1, (3 * real_len) // 4)
            prefix_ids = input_ids[i, :cut].cpu().tolist()

            if labels is not None:
                ref_tail_ids = labels[i, :real_len].cpu().tolist()
            else:
                ref_tail_ids = input_ids[i, cut:real_len].cpu().tolist()

            ref_tail_ids = [t for t in ref_tail_ids if t != -100]
            gen_len = len(ref_tail_ids)

            if gen_len <= 0:
                continue

            gen_full = greedy_generate(model, prefix_ids, gen_len, device)
            gen_tail = gen_full[len(prefix_ids):] if len(gen_full) > len(prefix_ids) else []

            pred_text = tokens_to_text(gen_tail, tokenizer)
            ref_text = tokens_to_text(ref_tail_ids, tokenizer)

            if len(ref_text.strip()) == 0:
                continue

            preds.append(pred_text)
            refs.append(ref_text)

            if printed < print_examples:
                print("\\n" + "=" * 50)
                print(f"ПРИМЕР ГЕНЕРАЦИИ LSTM {printed + 1}:")
                print("=" * 50)
                print(f"ПРЕФИКС: {tokens_to_text(prefix_ids, tokenizer)}")
                print(f"ОЖИДАЕМОЕ ПРОДОЛЖЕНИЕ: {ref_text}")
                print(f"СГЕНЕРИРОВАННОЕ ПРОДОЛЖЕНИЕ: {pred_text}")
                printed += 1

        if len(preds) >= max_examples:
            break

    rouge_scores = rouge.compute(predictions=preds, references=refs) if len(preds) > 0 else {}
    return rouge_scores, preds, refs