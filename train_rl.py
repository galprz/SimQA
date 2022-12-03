import argparse
import logging
import onmt.translate
import onmt.utils
import torch
import torch.nn as nn

from functools import partial
from metrics import BleuScore, CorrectAnswersScore, BleuAndStateScore
from onmt.utils.loss import NMTLossCompute
from pathlib import Path
from reward import blue_and_same_state_score
from tensorboardX import SummaryWriter
from trainer import SimStateScoreTrainer
from typing import Union

logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger()


device = "cuda" if torch.cuda.is_available() else "cpu"
device_id = 0 if torch.cuda.is_available() else -1
logger.info(f"Running on device=`{device}` (device_id={device_id})")

SAMPLE_SIZE = 8


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.generator.parameters():
        param.requires_grad = False


def load_model(model_path: Union[str, Path], embeddings_path: Union[str, Path]):
    emb_size = 200
    encoder_embeddings = onmt.modules.Embeddings(
        emb_size, len(src_vocab), word_padding_idx=src_padding, position_encoding=True, fix_word_vecs=True
    )
    encoder_embeddings.load_pretrained_vectors(embeddings_path)

    encoder = onmt.encoders.TransformerEncoder(
        num_layers=6,
        d_model=emb_size,
        max_relative_positions=100,
        embeddings=encoder_embeddings,
        d_ff=1,
        dropout=0.1,
        attention_dropout=0.1,
        heads=8,
    )

    decoder_embeddings = onmt.modules.Embeddings(
        emb_size, len(tgt_vocab), word_padding_idx=tgt_padding, position_encoding=True
    )
    decoder = onmt.decoders.TransformerDecoder(
        num_layers=6,
        d_model=emb_size,
        embeddings=decoder_embeddings,
        max_relative_positions=100,
        aan_useffn=False,
        full_context_alignment=True,
        alignment_layer=1,
        alignment_heads=1,
        heads=8,
        d_ff=1,
        copy_attn=False,
        self_attn_type="scaled-dot",
        dropout=0.1,
        attention_dropout=0.1,
    )

    model = onmt.models.model.NMTModel(encoder, decoder)

    # Specify the tgt word generator and loss computation module
    model.generator = nn.Sequential(nn.Linear(emb_size, len(tgt_vocab)), nn.LogSoftmax(dim=-1)).to(device)
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.generator.load_state_dict(checkpoint["generator"], strict=False)

    model.to(device)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Train RL")

    parser.add_argument("--name", type=str, default="")

    parser.add_argument("--train-version", type=str, default="v2", choices=("v1", "v2", "v3"))
    parser.add_argument("--valid-version", type=str, default="v3", choices=("v1", "v2", "v3"))
    parser.add_argument("--model-path", type=str, required=True)

    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--valid-batch-size", type=int, default=64)

    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--valid-steps", type=int, default=25)

    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-grad-norm", type=float, default=2.0)
    parser.add_argument("--reward-gamma", type=float, default=0.5)

    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)

    opts = parser.parse_args()

    writer = SummaryWriter(comment="-argmax")

    vocab_data_file = f"data/{opts.train_version}/processed/SimQA.vocab.pt"
    train_data_file = f"data/{opts.train_version}/processed/SimQA.train.0.pt"
    valid_data_file = f"data/{opts.valid_version}/processed/SimQA.valid.0.pt"
    model_path = f"{opts.model_path}"
    embeddings_path = f"models/{opts.train_version}/src.embeddings.pt"

    vocab_fields = torch.load(vocab_data_file)

    src_text_field = vocab_fields["src"].base_field
    src_vocab = src_text_field.vocab
    src_padding = src_vocab.stoi[src_text_field.pad_token]

    tgt_text_field = vocab_fields["tgt"].base_field
    tgt_vocab = tgt_text_field.vocab
    tgt_padding = tgt_vocab.stoi[tgt_text_field.pad_token]

    argmax_model = load_model(model_path=model_path, embeddings_path=embeddings_path)
    freeze_model(model=argmax_model)
    argmax_model.eval()

    model = load_model(model_path=model_path, embeddings_path=embeddings_path)

    loss = onmt.utils.loss.NMTLossCompute(
        criterion=nn.NLLLoss(ignore_index=tgt_padding, reduction="sum"), generator=model.generator
    )
    torch_optimizer = torch.optim.Adam(model.parameters(), lr=opts.learning_rate)
    optim = onmt.utils.optimizers.Optimizer(
        torch_optimizer, learning_rate=opts.learning_rate, max_grad_norm=opts.max_grad_norm
    )

    train_iter = onmt.inputters.inputter.DatasetLazyIter(
        dataset_paths=[train_data_file],
        fields=vocab_fields,
        batch_size=opts.train_batch_size,
        batch_size_multiple=1,
        batch_size_fn=None,
        device=device,
        is_train=True,
        repeat=True,
        pool_factor=8192,
    )

    valid_iter = onmt.inputters.inputter.DatasetLazyIter(
        dataset_paths=[valid_data_file],
        fields=vocab_fields,
        batch_size=opts.valid_batch_size,
        batch_size_multiple=1,
        batch_size_fn=None,
        device=device,
        is_train=False,
        repeat=False,
        pool_factor=8192,
    )
    report_manager = onmt.utils.ReportMgr(
        report_every=50,
        # tensorboard_writer=writer
    )

    src_reader = onmt.inputters.str2reader["text"]
    tgt_reader = onmt.inputters.str2reader["text"]
    scorer = onmt.translate.GNMTGlobalScorer(alpha=0.7, beta=0.0, length_penalty="avg", coverage_penalty="none")

    argmax_translator = onmt.translate.Translator(
        model=argmax_model,
        fields=vocab_fields,
        src_reader=src_reader,
        tgt_reader=tgt_reader,
        global_scorer=scorer,
        gpu=device_id,
        beam_size=1,
    )

    translator = onmt.translate.Translator(
        model=model,
        fields=vocab_fields,
        src_reader=src_reader,
        tgt_reader=tgt_reader,
        global_scorer=scorer,
        gpu=device_id,
        n_best=8,
        beam_size=32,
    )
    argmax_valid_translator = onmt.translate.Translator(
        model=model,
        fields=vocab_fields,
        src_reader=src_reader,
        tgt_reader=tgt_reader,
        global_scorer=scorer,
        gpu=device_id,
        beam_size=1,
    )
    argmax_builder = onmt.translate.TranslationBuilder(
        data=torch.load(train_data_file), fields=vocab_fields, has_tgt=True
    )

    valid_builder = onmt.translate.TranslationBuilder(
        data=torch.load(train_data_file), fields=vocab_fields, has_tgt=True
    )  # FIXME
    builder = onmt.translate.TranslationBuilder(
        data=torch.load(train_data_file), fields=vocab_fields, has_tgt=True, n_best=8
    )
    state_score_matric = BleuAndStateScore(tgt_vocab, 0.8)
    metrics = [BleuScore(), CorrectAnswersScore(tgt_vocab)]

    saved_model = Path.cwd().joinpath(f"checkpoints/RL")
    saved_model.mkdir(exist_ok=True, parents=True)
    saved_mode_model = saved_model.joinpath(f"{opts.name}")

    trainer = SimStateScoreTrainer(
        argmax_model=argmax_model,
        model=model,
        argmax_translator=argmax_translator,
        translator=translator,
        argmax_translation_builder=argmax_builder,
        translation_builder=builder,
        valid_translator=argmax_valid_translator,
        valid_builder=valid_builder,
        reward_function=partial(blue_and_same_state_score, gamma=opts.reward_gamma),
        model_saver=onmt.models.model_saver.ModelSaver(
            base_path=str(saved_mode_model), model=model, model_opt=vars(opts), fields=vocab_fields, optim=optim
        ),
        train_loss=loss,
        valid_loss=loss,
        optim=torch_optimizer,
        tgt_vocab=tgt_vocab,
        tgt_padding_token=tgt_padding,
        report_manager=report_manager,
        metrics=metrics,
        score_fn=state_score_matric,
    )
    print("this is when we test !")
    stats = trainer.test(test_iter=train_iter,
                         src_vocab=src_vocab)
    print(stats)
    # stats = trainer.train(
    #     train_iter=train_iter,
    #     src_vocab=src_vocab,
    #     train_steps=opts.train_steps,
    #     valid_iter=valid_iter,
    #     valid_steps=opts.valid_steps,
    #     save_checkpoint_steps=opts.save_every,
    # )
