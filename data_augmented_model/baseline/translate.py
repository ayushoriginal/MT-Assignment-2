import sys
import argparse
import torch as th
from tqdm import tqdm

from transformer import Transformer
from decoding import sample, greedy, beam_search
from training import load_data
from subwords import desegment


def get_args():
    parser = argparse.ArgumentParser("Translate with an MT model")
    # General params
    parser.add_argument("--src", type=str, default="en", choices=["en"])
    parser.add_argument("--tgt", type=str, default="af",
                        choices=["af", "ts", "nso"])
    parser.add_argument("--model-file", type=str,
                        default="model.pt", required=True)
    parser.add_argument("--input-file", type=str, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=15062019)
    # Model parameters
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    # Translation parameters
    parser.add_argument("--search", type=str, default="beam_search",
                        choices=["random", "greedy", "beam_search"])
    parser.add_argument("--beam-size", type=int, default=2)
    return parser.parse_args()


def move_to_device(tensors, device):
    return [tensor.to(device) for tensor in tensors]


def translate_sentence(
    model,
    sentence,
    beam_size=1,
    search="beam_search"
):
    # Convert string to indices
    src_tokens = [model.vocab[word] for word in sentence]
    # Decode
    with th.no_grad():
        if search == "random":
            out_tokens = sample(model, src_tokens)
        elif search == "greedy":
            out_tokens = greedy(model, src_tokens)
        else:
            out_tokens = beam_search(model, src_tokens, beam_size)
    # Convert back to strings
    return [model.vocab[tok] for tok in out_tokens]


def main():
    # Command line arguments
    args = get_args()
    # Fix seed for consistent sampling
    th.manual_seed(args.seed)
    # data
    vocab, _, _ = load_data(args.src, args.tgt)
    # Model
    model = Transformer(
        args.n_layers,
        args.embed_dim,
        args.hidden_dim,
        args.n_heads,
        vocab,
        args.dropout
    )
    if args.cuda:
        model = model.cuda()
    # Load existing model
    model.load_state_dict(th.load(args.model_file, map_location="cpu"))
    # Read from file/stdin
    if args.input_file is not None:
        input_stream = open(args.input_file, "r", encoding="utf-8")
    else:
        input_stream = sys.stdin
    # Write to file/stdout
    if args.output_file is not None:
        output_stream = open(args.output_file, "w", encoding="utf-8")
        # If we're printing to a file, display stats in stdout
        input_stream = tqdm(input_stream)
    else:
        output_stream = sys.stdout
    # Translate
    try:
        for line in input_stream:
            in_words = line.strip().split()
            out_words = translate_sentence(
                model,
                in_words,
                beam_size=args.beam_size,
                search=args.search,
            )
            print(desegment(out_words), file=output_stream)
            output_stream.flush()
    except KeyboardInterrupt:
        pass
    finally:
        input_stream.close()
        output_stream.close()


if __name__ == "__main__":
    main()
