
import os
import sys
import argparse
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, 'config', '.env')

load_dotenv(ENV_PATH)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src/data_prep')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src/model')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src/inference')))
from normalizer import Normalizer
from ngram_model import NGramModel
from predictor import Predictor


def _resolve_path(env_key, default_relative_path):
    configured_path = os.getenv(env_key, default_relative_path)
    if os.path.isabs(configured_path):
        return configured_path
    return os.path.join(BASE_DIR, configured_path)

def run_dataprep():
    input_folder = _resolve_path('INPUT_FOLDER', 'data/raw/train')
    output_file = _resolve_path('TOKEN_FILE', 'data/processed/train_tokens.txt')
    norm = Normalizer()
    texts = norm.load(input_folder)
    all_sentences = []
    for text in texts:
        text = norm.strip_gutenberg(text)
        text = norm.normalize(text)
        sentences = norm.sentence_tokenize(text)
        tokenized = [norm.word_tokenize(s) for s in sentences]
        all_sentences.extend(tokenized)
    norm.save(all_sentences, output_file)
    print(f"Data prep complete. Saved to {output_file}.")

def run_model():
    token_file = _resolve_path('TOKEN_FILE', 'data/processed/train_tokens.txt')
    model_path = _resolve_path('MODEL_PATH', 'data/model/model.json')
    vocab_path = _resolve_path('VOCAB_PATH', 'data/model/vocab.json')
    ngram_order = int(os.getenv('NGRAM_ORDER', 4))
    unk_threshold = int(os.getenv('UNK_THRESHOLD', 1))
    model = NGramModel(ngram_order=ngram_order, unk_threshold=unk_threshold)
    model.build_vocab(token_file)
    model.build_counts_and_probabilities(token_file)
    model.save_model(model_path)
    model.save_vocab(vocab_path)
    print(f"Model built and saved to {model_path} and {vocab_path}.")

def run_inference():
    model_path = _resolve_path('MODEL_PATH', 'data/model/model.json')
    vocab_path = _resolve_path('VOCAB_PATH', 'data/model/vocab.json')
    ngram_order = int(os.getenv('NGRAM_ORDER', 4))
    k = int(os.getenv('TOP_K', 3))
    model = NGramModel(ngram_order=ngram_order)
    model.load(model_path, vocab_path)
    normalizer = Normalizer()
    predictor = Predictor(model, normalizer)
    print("Type a sequence of words and get top-k predictions. Type 'quit' to exit.")
    try:
        while True:
            text = input('> ')
            if text.strip().lower() in {'quit', 'exit'}:
                print('Goodbye.')
                break
            predictions = predictor.predict_next(text, k)
            print(f"Predictions: {predictions}")
    except KeyboardInterrupt:
        print('\nGoodbye.')

def main():
    parser = argparse.ArgumentParser(description='NGram Predictor CLI')
    parser.add_argument('--step', type=str, required=True, choices=['dataprep', 'model', 'inference', 'all'], help='Pipeline step to run')
    args = parser.parse_args()

    if args.step == 'dataprep':
        run_dataprep()
    elif args.step == 'model':
        run_model()
    elif args.step == 'inference':
        run_inference()
    elif args.step == 'all':
        run_dataprep()
        run_model()
        run_inference()

if __name__ == "__main__":
    main()
#RUN Example
#python main.py --step all
#python main.py --step inference
#python main.py --step model
#python main.py --step dataprep
