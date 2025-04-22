"""
src/embeddings.py

Train a Word2Vec model on cleaned news text and generate average embeddings per row.

generate model:
python src/embeddings.py \
  --features    data/processed/features.csv \
  --model-path  models/w2v.model \
  --output-csv  data/processed/features_emb.csv

"""
import os
import argparse
import pandas as pd
import numpy as np
from gensim.models import Word2Vec


def train_word2vec(corpus, vector_size=100, window=5, min_count=3, workers=4, model_path="models/w2v.model"):
    """
    Train and save a Word2Vec model on the provided corpus (list of token lists).
    """
    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=10
    )
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"[EMB] Trained Word2Vec model and saved to {model_path}")
    return model


def get_average_vector(tokens, model):
    """
    Compute the average vector for a list of tokens using the Word2Vec model.
    Returns a numpy array of shape (vector_size,).
    """
    vectors = [model.wv[t] for t in tokens if t in model.wv]
    if not vectors:
        return np.zeros(model.vector_size, dtype=float)
    return np.mean(vectors, axis=0)


def add_embeddings(df, model):
    """
    Take a DataFrame with a 'text_raw' column, tokenize, compute avg embeddings, and return augmented DataFrame.
    """
    # Ensure no NaN in text_raw
    df['text_raw'] = df['text_raw'].fillna("")
    # Tokenize by whitespace
    df['tokens'] = df['text_raw'].apply(lambda txt: txt.split())
    # Compute average vectors
    embedding_matrix = df['tokens'].apply(lambda toks: get_average_vector(toks, model))
    vector_size = model.vector_size
    emb_cols = [f'emb_{i}' for i in range(vector_size)]
    emb_df = pd.DataFrame(embedding_matrix.tolist(), columns=emb_cols)
    result = pd.concat([df.reset_index(drop=True), emb_df], axis=1)
    return result.drop(columns=['tokens'])


def main():
    parser = argparse.ArgumentParser(description="Train Word2Vec and generate embeddings")
    parser.add_argument('--features', required=True,
                        help='Path to preprocessed features CSV')
    parser.add_argument('--model-path', default='../models/w2v.model',
                        help='Path to save/load Word2Vec model')
    parser.add_argument('--output-csv', default='../data/processed/features_emb.csv',
                        help='Output CSV path with embeddings')
    parser.add_argument('--vector-size', type=int, default=100)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--min-count', type=int, default=3)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    # Read features, ensure text_raw is string
    df = pd.read_csv(args.features)
    df['text_raw'] = df['text_raw'].fillna("")

    # Prepare corpus for training
    corpus = df['text_raw'].apply(lambda txt: txt.split()).tolist()

    # Train model if not exists
    if not os.path.exists(args.model_path):
        model = train_word2vec(
            corpus,
            vector_size=args.vector_size,
            window=args.window,
            min_count=args.min_count,
            workers=args.workers,
            model_path=args.model_path
        )
    else:
        model = Word2Vec.load(args.model_path)
        print(f"[EMB] Loaded existing model from {args.model_path}")

    # Generate and save embeddings
    df_emb = add_embeddings(df, model)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df_emb.to_csv(args.output_csv, index=False)
    print(f"[EMB] Wrote features with embeddings to {args.output_csv}")


if __name__ == '__main__':
    main()
