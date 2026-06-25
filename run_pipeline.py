import random
from src import meta_model, bert_classifier, llm_classifier
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import sys
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
from string import punctuation

def predict_meta(df, bert_model, logistic_model, scaler, return_proba=False):
    bert_model.eval()
    texts = df["text"].astype(str).tolist()
    with torch.no_grad():
        inference_dataset = bert_classifier.ReviewDataset(texts, [0] * len(texts))
        inference_loader = DataLoader(
            inference_dataset,
            batch_size=bert_classifier.BATCH_SIZE,
            shuffle=False,
            pin_memory=False,
            num_workers=0,
            collate_fn=bert_classifier.collate_batch
        )
        bert_embeddings, _ = meta_model.get_bert_emb(bert_model, inference_loader)
        custom_features = meta_model.get_custom_features(df)
        X_meta = np.hstack((bert_embeddings, custom_features))
        X_meta_scaled = scaler.transform(X_meta)
        predictions = logistic_model.predict(X_meta_scaled)
        if return_proba:
            proba = logistic_model.predict_proba(X_meta_scaled)[:, 1]
            return predictions, proba
    return predictions

def predict_bert(texts, bert_model, return_proba=False):
    bert_model.eval()
    with torch.no_grad():
        inference_dataset = bert_classifier.ReviewDataset(texts, [0] * len(texts))
        inference_loader = DataLoader(
            inference_dataset,
            batch_size=bert_classifier.BATCH_SIZE,
            shuffle=False,
            pin_memory=False,
            num_workers=0,
            collate_fn=bert_classifier.collate_batch
        )
        predictions, scores = [], []
        for batch in inference_loader:
            input_ids = batch['input_ids'].to(bert_classifier.DEVICE)
            attention_mask = batch['attention_mask'].to(bert_classifier.DEVICE)
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            predictions.extend(preds.tolist())
            scores.extend(probs[:, 1].cpu().numpy().tolist())
    if return_proba:
        return predictions, np.array(scores)
    return predictions

def explain_attention(text, bert_model, tokenizer, top_k=8):
    """Возвращает attention скоры для каждого слова"""
    bert_model.eval()
    enc = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=bert_classifier.MAX_LEN
    ).to(bert_classifier.DEVICE)
    with torch.no_grad():
        out = bert_model(**enc, output_attentions=True)
    label = int(torch.argmax(out.logits, dim=1).item())
    last = out.attentions[-1][0]
    cls_attn = last[:, 0, :].mean(0).cpu()
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    punc_set = set(punctuation)
    punc_set.add("’")
    words, scores = [], []
    for tok, sc in zip(tokens, cls_attn.tolist()):
        if tok in tokenizer.all_special_tokens:
            continue
        if tok.startswith("##") and words:
            words[-1] += tok[2:]
            scores[-1] += sc
        elif set(tok) <= punc_set or tok.isspace():
            continue
        else:
            words.append(tok)
            scores.append(sc)
    top_words = sorted(zip(words, scores), key=lambda w: w[1], reverse=True)[:top_k]
    return label, top_words


KEEP_COLS = ["text", "label", "rating", "category"]
LLM_TEST_SAMPLE = 20


def _normalize(df):
    """Приводит датасет к одинаковому виду"""
    for col in KEEP_COLS:
        if col not in df.columns:
            df[col] = 0 if col in ("label", "rating") else ""
    df = df[KEEP_COLS].copy()
    df["text"]   = df["text"].astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)
    df = df[df["label"].notna()]
    return df.reset_index(drop=True)


def _balance(df, random_state=42):
    """Сводит классы к 50/50"""
    counts = df["label"].value_counts()
    if len(counts) < 2:
        return df.iloc[0:0]
    n = int(counts.min())
    parts = [grp.sample(n, random_state=random_state) for _, grp in df.groupby("label")]
    return pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)


def load_test_datasets(data_dir):
    """Грузит 3 тестовых датасета и переименовывает колонки"""
    random.seed(42)

    df_pseudo = pd.read_csv(
        data_dir / 'pseudo_labeled_amazon_reviews.csv',
        skiprows=lambda i: (i > 0) and (random.random() > 0.1)
    ).fillna("")

    df_llm = pd.read_csv(data_dir / 'amazon_reviews_llm_annotated.csv').fillna("")
    df_llm = df_llm.rename(columns={"review": "text"})
    df_llm["label"] = 0

    df_fake = pd.read_csv(data_dir / 'fake reviews dataset.csv').fillna("")
    df_fake = df_fake.rename(columns={"text_": "text"})
    df_fake["label"] = df_fake["label"].map({"CG": 0, "OR": 1})
    # у категорий _5 вконце
    df_fake["category"] = df_fake["category"].astype(str).str.replace(r"_\d+$", "", regex=True)

    train_df, test_df, _, _ = bert_classifier.build_train_test()
    seen = set(train_df["text"].astype(str)) | set(test_df["text"].astype(str))

    datasets = {
        "pseudo_labeled": _normalize(df_pseudo),
        "llm_annotated":  _normalize(df_llm),
        "fake_reviews":   _normalize(df_fake),
    }

    for name in ("pseudo_labeled", "llm_annotated"):
        df = datasets[name]
        before = len(df)
        datasets[name] = df[~df["text"].isin(seen)].reset_index(drop=True)
        print(f"{name}: убрано {before - len(datasets[name])} обучающих строк, осталось {len(datasets[name])}")

    for name in list(datasets):
        before = len(datasets[name])
        datasets[name] = _balance(datasets[name])
        print(f"{name}: баланс 50/50 -> {len(datasets[name])} (было {before})")

    return {name: df for name, df in datasets.items() if len(df) > 0}

def evaluate(name, df, mode, bert_model, logistic_model=None, scaler=None, tokenizer=None):
    """Прогоняет модель по одному датасету и считает метрики."""
    y_true = df["label"].to_numpy()
    if mode == "meta":
        y_pred, y_score = predict_meta(df, bert_model, logistic_model, scaler, return_proba=True)
    elif mode == "base":
        X = tokenizer.transform(df["text"].to_numpy())
        y_pred = logistic_model.predict(X)
        y_score = logistic_model.predict_proba(X)[:, 1]
    elif mode == "llm":
        preds = llm_classifier.classify_reviews(df["text"].tolist(), df["rating"].tolist())
        mask = np.array([p is not None for p in preds])
        y_true = y_true[mask]
        y_pred = np.array([p for p in preds if p is not None])
        y_score = y_pred
    else:
        y_pred, y_score = predict_bert(df["text"].tolist(), bert_model, return_proba=True)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    pr_auc = average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    return {"dataset": name, "n": len(df), "accuracy": acc,
            "precision": p, "recall": r, "f1": f1, "roc_auc": auc, "pr_auc": pr_auc}

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / 'data/raw'
    args = sys.argv[1:]

    if args[:1] == ["run"]:
        if len(args) < 2:
            print("использование: run_pipeline.py run <путь к файлу с отзывами>")
            sys.exit(1)
        with open(args[1], 'r', encoding='utf-8') as f:
            reviews = [line.strip() for line in f if line.strip()]
        data = pd.read_csv(args[1]).fillna("")
        data.rename(columns={"review": "text"}, inplace=True)
        tokenizer = BertTokenizerFast.from_pretrained(meta_model.model_path)
        bert_model = BertForSequenceClassification.from_pretrained(
            meta_model.model_path, attn_implementation="eager"
        )
        bert_model.to(bert_classifier.DEVICE)
        imported_model = joblib.load(bert_classifier.MODELS_DIR / 'meta_model.pkl')
        logistic_model = imported_model['meta_model']
        scaler = imported_model['scaler']

        predictions = predict_meta(data, bert_model, logistic_model, scaler)
        print("\n---META---\n")
        print(predictions)
        print(f"0: {predictions.tolist().count(0)}, 1: {predictions.tolist().count(1)}")
        print("\n---BERT---\n")
        bert_predictions = predict_bert(data['text'].tolist(), bert_model)
        print(bert_predictions)
        print(f"0: {bert_predictions.count(0)}, 1: {bert_predictions.count(1)}")
        print("\n---\n веса:")
        print(logistic_model.coef_[0])
        print("\n---байас:\n")
        print(logistic_model.intercept_)

        # print("\n---объяснение---")
        # for text, meta_pred in zip(data['text'].astype(str).tolist(), predictions):
        #     label, top_words = explain_attention(text, bert_model, tokenizer)
        #     words_str = ", ".join(f"{w} ({s:.3f})" for w, s in top_words)
        #     print(f"\n[мета: {bert_classifier.id2label[int(meta_pred)]} | BERT: {bert_classifier.id2label[label]}] {text[:80]}")
        #     print(f"  {words_str}")

    elif args[:1] == ["test"]:
        mode = args[1] if len(args) > 1 else "meta"
        if mode not in ("meta", "bert", "base", "bert_base", "llm"):
            print("использование: run_pipeline.py test [meta|bert|base|bert_base|llm]")
            sys.exit(1)

        datasets = load_test_datasets(data_dir)
        if mode == "llm":
            datasets = {
                name: df.sample(min(len(df), LLM_TEST_SAMPLE), random_state=42).reset_index(drop=True)
                for name, df in datasets.items()
            }

        bert_model = tokenizer = logistic_model = scaler = None
        if mode in ("meta", "bert"):
            tokenizer = BertTokenizerFast.from_pretrained(meta_model.model_path)
            bert_model = BertForSequenceClassification.from_pretrained(meta_model.model_path)
            bert_model.to(bert_classifier.DEVICE)
        elif mode == "bert_base":
            tokenizer = BertTokenizerFast.from_pretrained(bert_classifier.MODEL_NAME)
            bert_model = BertForSequenceClassification.from_pretrained(
                bert_classifier.MODEL_NAME, num_labels=2
            )
            bert_model.to(bert_classifier.DEVICE)

        if mode == "meta":
            imported_model = joblib.load(bert_classifier.MODELS_DIR / 'meta_model.pkl')
            logistic_model = imported_model['meta_model']
            scaler = imported_model['scaler']
        elif mode == "base":
            model_type = args[2] if len(args) > 2 else "TF_IDF"
            imported_model = joblib.load(bert_classifier.MODELS_DIR / f'{model_type}.pkl')
            logistic_model = imported_model['model']
            tokenizer = imported_model['vectorizer']
        rows = [
            evaluate(name, df, mode, bert_model, logistic_model, scaler, tokenizer)
            for name, df in datasets.items()
        ]
        table = pd.DataFrame(rows).set_index("dataset")
        print(f"\n=== Результаты ({mode}) ===")
        print(table.to_string(float_format=lambda x: f"{x:.3f}"))

    else:
        print("нужен флаг: run | test")
