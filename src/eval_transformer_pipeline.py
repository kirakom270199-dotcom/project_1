import torch
from tqdm import tqdm
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_transformer_model(model_name="distilgpt2"):
    """Загрузка предобученной трансформерной модели"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.eval()
        print("✅ Transformer модель успешно загружена!")
        return tokenizer, model

    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return None, None


@torch.no_grad()
def transformers_generate(model, tokenizer, text_prefix, max_new_tokens=50, temperature=0.7):
    """Генерация продолжения текста с помощью предобученного трансформера"""
    inputs = tokenizer.encode(text_prefix, return_tensors="pt")

    outputs = model.generate(
        inputs,
        max_length=inputs.shape[1] + max_new_tokens,
        num_return_sequences=1,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_part = generated_text[len(text_prefix):].strip()

    return generated_part


@torch.no_grad()
def evaluate_transformer_generation(loader, lstm_tokenizer, transformer_tokenizer, transformer_model, max_examples=200,
                                    print_examples=5):
    """Валидация предобученного трансформера"""
    print("\\n🧪 Запускаем валидацию предобученного трансформера...")

    rouge = evaluate.load("rouge")
    preds, refs = [], []
    printed = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="Transformers Validation")):
        if len(preds) >= max_examples:
            break

        input_ids = batch['input_ids']
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
            if real_len < 10:
                continue

            cut = max(1, (3 * real_len) // 4)
            prefix_ids = input_ids[i, :cut]

            if labels is not None:
                ref_tail_ids = labels[i, :real_len]
            else:
                ref_tail_ids = input_ids[i, cut:real_len]

            ref_tail_ids = ref_tail_ids[ref_tail_ids != -100]
            gen_len = len(ref_tail_ids)

            if gen_len <= 0:
                continue

            prefix_text = lstm_tokenizer.decode(prefix_ids.cpu().tolist(), skip_special_tokens=True)
            ref_text = lstm_tokenizer.decode(ref_tail_ids.cpu().tolist(), skip_special_tokens=True)

            if len(ref_text.strip()) == 0:
                continue

            try:
                pred_text = transformers_generate(transformer_model, transformer_tokenizer, prefix_text,
                                                  max_new_tokens=gen_len)

                preds.append(pred_text)
                refs.append(ref_text)

                if printed < print_examples:
                    print("\\n" + "=" * 60)
                    print(f"ПРИМЕР ГЕНЕРАЦИИ TRANSFORMER {printed + 1}:")
                    print("=" * 60)
                    print(f"ПРЕФИКС: {prefix_text}")
                    print(f"ОЖИДАЕМОЕ ПРОДОЛЖЕНИЕ: {ref_text}")
                    print(f"СГЕНЕРИРОВАННОЕ ПРОДОЛЖЕНИЕ: {pred_text}")
                    printed += 1

            except Exception as e:
                print(f"Ошибка генерации: {e}")
                continue

    if len(preds) > 0:
        rouge_scores = rouge.compute(predictions=preds, references=refs)
        print(f"\\n✅ Проверено примеров: {len(preds)}")
        return rouge_scores, preds, refs
    else:
        print("❌ Не удалось сгенерировать примеры для валидации")
        return {}, [], []


def compare_models(lstm_results, transformer_results, lstm_params, transformer_params):
    """Сравнение результатов двух моделей"""
    print("\\n" + "=" * 80)
    print("СРАВНЕНИЕ МОДЕЛЕЙ: LSTM vs TRANSFORMER")
    print("=" * 80)

    lstm_rouge = lstm_results.get('rouge', {})
    transformer_rouge = transformer_results.get('rouge', {})

    if lstm_rouge and transformer_rouge:
        print("\\n📊 ROUGE МЕТРИКИ:")
        print(f"{'Метрика':<10} {'LSTM':<8} {'Transformer':<12} {'Разница':<10}")
        print("-" * 45)

        for metric in ['rouge1', 'rouge2', 'rougeL']:
            lstm_score = lstm_rouge.get(metric, 0)
            transformer_score = transformer_rouge.get(metric, 0)
            difference = transformer_score - lstm_score
            print(f"{metric:<10} {lstm_score:.4f}    {transformer_score:.4f}       {difference:+.4f}")

    print(f"\\n📏 РАЗМЕРЫ МОДЕЛЕЙ:")
    print(f"LSTM: {lstm_params:,} параметров")
    print(f"Transformer: {transformer_params:,} параметров")
    print(f"Отношение: {transformer_params / lstm_params:.1f}x")

    return {
        'lstm': {'rouge': lstm_rouge, 'params': lstm_params},
        'transformer': {'rouge': transformer_rouge, 'params': transformer_params}
    }