import argparse
import logging
import onmt.translate
import onmt.utils.loss
import torch
import torch.nn as nn

from metrics import BleuScore, CorrectAnswersScore
from pathlib import Path
from tensorboardX import SummaryWriter
from trainer import SimMLETrainer

logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger()


device = "cuda" if torch.cuda.is_available() else "cpu"
device_id = 0 if torch.cuda.is_available() else -1
logger.info(f"Running on device=`{device}` (device_id={device_id})")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="Train MLE")

    parser.add_argument("--mode", type=str, default="v1", choices=("v1, v2"))

    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--valid-batch-size", type=int, default=16)

    parser.add_argument("--train-steps", type=int, default=10000)
    parser.add_argument("--valid-steps", type=int, default=100)

    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-grad-norm", type=float, default=2.0)

    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)

    opts = parser.parse_args()

    vocab_data_file = f"data/{opts.mode}/processed/SimQA.vocab.pt"
    train_data_file = f"data/{opts.mode}/processed/SimQA.train.0.pt"
    valid_data_file = f"data/{opts.mode}/processed/SimQA.valid.0.pt"
    embeddings_file = f"models/{opts.mode}/src.embeddings.pt"

    vocab_fields = torch.load(vocab_data_file)

    src_text_field = vocab_fields["src"].base_field
    src_vocab = src_text_field.vocab
    src_padding = src_vocab.stoi[src_text_field.pad_token]

    tgt_text_field = vocab_fields["tgt"].base_field
    tgt_vocab = tgt_text_field.vocab
    tgt_padding = tgt_vocab.stoi[tgt_text_field.pad_token]

    emb_size = 200
    encoder_embeddings = onmt.modules.Embeddings(
        emb_size, len(src_vocab), word_padding_idx=src_padding, position_encoding=True, fix_word_vecs=True
    )
    encoder_embeddings.load_pretrained_vectors(embeddings_file)

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

    model = onmt.models.model.NMTModel(encoder, decoder).to(device)

    # Specify the tgt word generator and loss computation module
    model.generator = nn.Sequential(nn.Linear(emb_size, len(tgt_vocab)), nn.LogSoftmax(dim=-1)).to(device)

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

    saved_model = Path.cwd().joinpath(f"checkpoints")
    saved_model.mkdir(exist_ok=True, parents=True)
    saved_mode_model = saved_model.joinpath(f"{opts.mode}")

    trainer = SimMLETrainer(
        model=model,
        model_saver=onmt.models.model_saver.ModelSaver(
            base_path=str(saved_mode_model), model=model, model_opt=vars(opts), fields=vocab_fields, optim=optim
        ),
        train_loss=loss,
        valid_loss=loss,
        optim=optim,
        report_manager=onmt.utils.ReportMgr(
            report_every=opts.report_every, tensorboard_writer=SummaryWriter(comment="-argmax")
        ),
        metrics=[BleuScore(), CorrectAnswersScore(tgt_vocab)],
        translator=onmt.translate.Translator(
            model=model,
            fields=vocab_fields,
            src_reader=onmt.inputters.str2reader["text"],
            tgt_reader=onmt.inputters.str2reader["text"],
            global_scorer=onmt.translate.GNMTGlobalScorer(
                alpha=0.7, beta=0.0, length_penalty="avg", coverage_penalty="none"
            ),
            gpu=device_id,
        ),
        translation_builder=onmt.translate.TranslationBuilder(
            data=torch.load(valid_data_file), fields=vocab_fields, has_tgt=True
        ),
    )
    stats, metrics = trainer.train(
        train_iter=train_iter,
        src_vocab=src_vocab,
        train_steps=opts.train_steps,
        valid_iter=valid_iter,
        valid_steps=opts.valid_steps,
        save_checkpoint_steps=opts.save_every,
    )
