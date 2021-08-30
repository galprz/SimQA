import torch
import torch.nn as nn
import logging
from tensorboardX import SummaryWriter
import onmt.opts
import onmt.translate
import onmt.utils.loss
from metrics import BleuScore, CorrectAnswersScore
from trainer import SimMLETrainer
from itertools import chain

if __name__ == "__main__":
    mode = "v1"
    writer = SummaryWriter(comment="-argmax")
    fmt = "%(asctime)-15s %(levelname)s %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO)
    scenario_number = 8
    vocab_fields = torch.load(f"processed/SimQA.vocab.pt")

    src_text_field = vocab_fields["src"].base_field
    src_vocab = src_text_field.vocab
    src_padding = src_vocab.stoi[src_text_field.pad_token]

    tgt_text_field = vocab_fields["tgt"].base_field
    tgt_vocab = tgt_text_field.vocab
    tgt_padding = tgt_vocab.stoi[tgt_text_field.pad_token]

    emb_size = 200
    rnn_size = 200
    encoder_embeddings = onmt.modules.Embeddings(
        emb_size,
        len(src_vocab),
        word_padding_idx=src_padding,
        position_encoding=True,
        fix_word_vecs=True,
    )
    encoder_embeddings.load_pretrained_vectors(f"src.embeddings.pt")

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = onmt.models.model.NMTModel(encoder, decoder)
    model.to(device)
    logging.getLogger().info(device)
    # Specify the tgt word generator and loss computation module
    model.generator = nn.Sequential(nn.Linear(emb_size, len(tgt_vocab)), nn.LogSoftmax(dim=-1)).to(device)

    loss = onmt.utils.loss.NMTLossCompute(
        criterion=nn.NLLLoss(ignore_index=tgt_padding, reduction="sum"), generator=model.generator
    )

    lr = 0.001
    torch_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optim = onmt.utils.optimizers.Optimizer(torch_optimizer, learning_rate=lr, max_grad_norm=2)

    train_data_file = f"processed/SimQA.train.0.pt"
    valid_data_file = f"processed/SimQA.valid.0.pt"
    train_iter = onmt.inputters.inputter.DatasetLazyIter(
        dataset_paths=[train_data_file],
        fields=vocab_fields,
        batch_size=32,
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
        batch_size=16,
        batch_size_multiple=1,
        batch_size_fn=None,
        device=device,
        is_train=False,
        repeat=False,
        pool_factor=8192,
    )
    report_manager = onmt.utils.ReportMgr(report_every=100, tensorboard_writer=writer)

    src_reader = onmt.inputters.str2reader["text"]
    tgt_reader = onmt.inputters.str2reader["text"]
    scorer = onmt.translate.GNMTGlobalScorer(alpha=0.7, beta=0.0, length_penalty="avg", coverage_penalty="none")
    gpu = 0 if torch.cuda.is_available() else -1
    translator = onmt.translate.Translator(
        model=model, fields=vocab_fields, src_reader=src_reader, tgt_reader=tgt_reader, global_scorer=scorer, gpu=gpu
    )
    builder = onmt.translate.TranslationBuilder(data=torch.load(valid_data_file), fields=vocab_fields, has_tgt=True)

    metrics = [BleuScore(), CorrectAnswersScore(tgt_vocab)]
    model_saver = onmt.models.model_saver.ModelSaver("model", model, None, vocab_fields, optim)
    trainer = SimMLETrainer(
        model=model,
        model_saver=model_saver,
        train_loss=loss,
        valid_loss=loss,
        optim=optim,
        report_manager=report_manager,
        metrics=metrics,
        translator=translator,
        translation_builder=builder,
    )
    stats, metrics = trainer.train(
        train_iter=train_iter,
        src_vocab=src_vocab,
        train_steps=10000,
        valid_iter=valid_iter,
        valid_steps=100,
        save_checkpoint_steps=1000,
    )
