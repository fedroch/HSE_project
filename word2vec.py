import re
import numpy as np
import joblib
from pathlib import Path
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from src.bert_classifier import build_train_test

MODELS_DIR = Path(__file__).parent.parent / 'models'
MODELS_DIR.mkdir(exist_ok=True)

EMBED_DIM = 300
WINDOW    = 5
MIN_COUNT = 5
NEG       = 5
EPOCHS    = 20
WORKERS   = 4

def tokenize(text):
    return re.findall(r'\b\w+\b', str(text).lower())

class MeanEmbeddingVectorizer:
    def __init__(self, wv):
        self.wv = wv
        self.dim = wv.vector_size

    def transform(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            vecs = [self.wv[w] for w in tokenize(text) if w in self.wv]
            if vecs:
                out[i] = np.mean(vecs, axis=0)
        return out

if __name__ == "__main__":
    from src.word2vec import MeanEmbeddingVectorizer
    X_train_raw, X_test_raw, y_train, y_test = build_train_test()
    print("Токенизация")
    train_sentences = [tokenize(t) for t in X_train_raw["text"].tolist()]
    print("Обучение word2vec")
    w2v = Word2Vec(
        sentences=train_sentences,
        vector_size=EMBED_DIM,
        window=WINDOW,
        min_count=MIN_COUNT,
        sg=1,
        negative=NEG,
        epochs=EPOCHS,
        workers=WORKERS,
        seed=42,
    )
    print(f"Размер словаря: {len(w2v.wv):,}")
    vectorizer = MeanEmbeddingVectorizer(w2v.wv)
    print("Векторизация")
    X_train = vectorizer.transform(X_train_raw["text"].tolist())
    X_test = vectorizer.transform(X_test_raw["text"].tolist())
    print("Обучение logisticRegression")
    logistic_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    logistic_model.fit(X_train, y_train)
    predictions = logistic_model.predict(X_test)
    joblib.dump({'model': logistic_model, 'vectorizer': vectorizer},
                MODELS_DIR / 'word2vec.pkl')
    print(f"Модель сохранена в {MODELS_DIR / 'word2vec.pkl'}")
    print(f"\nОбщая точность: {accuracy_score(y_test, predictions):.2%}")
    print("\nПодробный отчет:")
    print(classification_report(y_test, predictions))
