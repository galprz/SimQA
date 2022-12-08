import random
import time
from heapq import heappush, nlargest
from tqdm import tqdm
from onmt.trainer import Trainer
from onmt.translate import GreedySearch
from onmt.utils.logging import logger
import onmt
import torch
import traceback
import numpy as np

from grammer.tokenizer import SimCodeTokenizer
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from reward import state_transition_reward


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)


class SimMLETrainer(Trainer):
    def __init__(
        self,
        model,
        train_loss,
        valid_loss,
        optim,
        translator,
        translation_builder,
        score_fn=None,
        trunc_size=0,
        shard_size=32,
        norm_method="sents",
        accum_count=[1],
        accum_steps=[0],
        n_gpu=1,
        gpu_rank=1,
        gpu_verbose_level=0,
        report_manager=None,
        with_align=False,
        model_saver=None,
        average_decay=0,
        average_every=1,
        model_dtype="fp32",
        earlystopper=None,
        dropout=[0.3],
        dropout_steps=[0],
        source_noise=None,
        metrics=None,
    ):
        super().__init__(
            model,
            train_loss,
            valid_loss,
            optim,
            trunc_size,
            shard_size,
            norm_method,
            accum_count,
            accum_steps,
            n_gpu,
            gpu_rank,
            gpu_verbose_level,
            report_manager,
            with_align,
            model_saver,
            average_decay,
            average_every,
            model_dtype,
            earlystopper,
            dropout,
            dropout_steps,
            source_noise,
        )
        self.metrics = [] if metrics is None else metrics
        self.translator = translator
        self.translation_builder = translation_builder
        self.score_fn = score_fn
        self.lookahead = self.model_saver.model_opt["lookahead"]

    def train(
        self,
        train_iter,
        train_steps,
        src_vocab=None,
        save_checkpoint_steps=5000,
        valid_iter=None,
        valid_steps=10000,
        stats_cls=None,
    ):
        """
        The main training loop by iterating over `train_iter` and possibly
        running validation on `valid_iter`.

        Args:
            train_iter: A generator that returns the next training batch.
            train_steps: Run training for this many iterations.
            save_checkpoint_steps: Save a checkpoint every this many
              iterations.
            valid_iter: A generator that returns the next validation batch.
            valid_steps: Run evaluation every this many iterations.

        Returns:
            The gathered statistics.
        """
        if not self.metrics:
            logger.info("No specific success metric mentioned")

        if valid_iter is None:
            logger.info("Start training loop without validation...")
        else:
            logger.info("Start training loop and validate every %d steps...", valid_steps)

        if stats_cls is None:
            total_stats = onmt.utils.Statistics()
            report_stats = onmt.utils.Statistics()
        else:
            total_stats = stats_cls()
            report_stats = stats_cls()

        self._start_report_manager(start_time=total_stats.start_time)
        all_metrics = {}
        for i, (batches, normalization) in enumerate(self._accum_batches(train_iter)):
            step = self.optim.training_step
            # UPDATE DROPOUT
            self._maybe_update_dropout(step)

            if self.gpu_verbose_level > 1:
                logger.info("GpuRank %d: index: %d", self.gpu_rank, i)
            if self.gpu_verbose_level > 0:
                logger.info(
                    "GpuRank %d: reduce_counter: %d \
                                    n_minibatch %d"
                    % (self.gpu_rank, i + 1, len(batches))
                )

            if self.n_gpu > 1:
                normalization = sum(onmt.utils.distributed.all_gather_list(normalization))

            self._gradient_accumulation(batches, normalization, total_stats, report_stats)

            if self.average_decay > 0 and i % self.average_every == 0:
                self._update_average(step)

            report_stats = self._maybe_report_training(step, train_steps, self.optim.learning_rate(), report_stats)

            if valid_iter is not None and step % valid_steps == 0:
                if self.gpu_verbose_level > 0:
                    logger.info("GpuRank %d: validate step %d" % (self.gpu_rank, step))
                valid_stats = self.validate(
                    valid_iter, src_vocab, moving_average=self.moving_average, stats_cls=stats_cls
                )
                all_metrics[str(i)] = []
                for metric in self.metrics:
                    all_metrics[str(i)].append(metric.eval())
                if self.gpu_verbose_level > 0:
                    logger.info(
                        "GpuRank %d: gather valid stat \
                                        step %d"
                        % (self.gpu_rank, step)
                    )
                valid_stats = self._maybe_gather_stats(valid_stats)
                if self.gpu_verbose_level > 0:
                    logger.info("GpuRank %d: report stat step %d" % (self.gpu_rank, step))
                self._report_step(self.optim.learning_rate(), step, valid_stats=valid_stats)
                # Run patience mechanism
                if self.earlystopper is not None:
                    self.earlystopper(valid_stats, step)
                    # If the patience has reached the limit, stop training
                    if self.earlystopper.has_stopped():
                        break

            if self.model_saver is not None and (save_checkpoint_steps != 0 and step % save_checkpoint_steps == 0):
                self.model_saver.save(step, moving_average=self.moving_average)

            if train_steps > 0 and step >= train_steps:
                break

        if self.model_saver is not None:
            self.model_saver.save(step, moving_average=self.moving_average)
        return total_stats, all_metrics

    def validate(self, valid_iter, src_vocab, moving_average=None, stats_cls=None):
        """Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        if self.lookahead:
            self.optim._optimizer._backup_and_load_cache()

        valid_model = self.model
        if moving_average:
            # swap model params w/ moving average
            # (and keep the original parameters)
            model_params_data = []
            for avg, param in zip(self.moving_average, valid_model.parameters()):
                model_params_data.append(param.data)
                param.data = avg.data.half() if self.optim._fp16 == "legacy" else avg.data

        # Set model in validating mode.
        logger.info("MLE pre-trained model validation:")
        valid_model.eval()

        with torch.no_grad():
            if stats_cls is None:
                stats = onmt.utils.Statistics()
            else:
                stats = stats_cls()
            # reset metric
            for metric in self.metrics:
                metric.reset()

            for batch in valid_iter:
                src, src_lengths = batch.src if isinstance(batch.src, tuple) else (batch.src, None)
                tgt = batch.tgt

                # F-prop through the model.
                outputs, attns = valid_model(src, tgt, src_lengths, with_align=self.with_align)

                # Compute loss.
                _, batch_stats = self.valid_loss(batch, outputs, attns)

                trans_batch = self.translator.translate_batch(batch=batch, src_vocabs=[src_vocab], attn_debug=False)
                translations = self.translation_builder.from_batch(trans_batch)
                targets = []
                preds = []
                for trans in translations:
                    max_score = 0
                    pred_sent = trans.pred_sents[0]
                    if self.score_fn is not None:
                        for pred_seq in trans.pred_sents:
                            self.score_fn.reset()
                            self.score_fn.update([pred_seq], [trans.gold_sent])
                            score = self.score_fn.eval()
                            if score > max_score:
                                pred_sent = pred_seq
                                max_score = score
                    preds.append(pred_sent)
                    targets.append(trans.gold_sent)

                for metric in self.metrics:
                    metric.update(preds, targets)

                # Update statistics.
                stats.update(batch_stats)
            metrics_txt = ",".join(f"{metric}" for metric in self.metrics)
            logger.info(f"Validation metrics: {metrics_txt}")
        if moving_average:
            for param_data, param in zip(model_params_data, self.model.parameters()):
                param.data = param_data

        # Set model back to training mode.
        valid_model.train()

        if self.lookahead:
            self.optim._optimizer._clear_and_load_backup()

        return stats

    def _gradient_accumulation(self, true_batches, normalization, total_stats, report_stats):
        if self.accum_count > 1:
            self.optim.zero_grad()

        for k, batch in enumerate(true_batches):
            target_size = batch.tgt.size(0)
            # Truncated BPTT: reminder not compatible with accum > 1
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            batch = self.maybe_noise_source(batch)

            src, src_lengths = batch.src if isinstance(batch.src, tuple) else (batch.src, None)
            if src_lengths is not None:
                report_stats.n_src_words += src_lengths.sum().item()

            tgt_outer = batch.tgt

            bptt = False
            for j in range(0, target_size - 1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j : j + trunc_size]

                # 2. F-prop all but generator.
                if self.accum_count == 1:
                    self.optim.zero_grad()

                outputs, attns = self.model(src, tgt, src_lengths, bptt=bptt, with_align=self.with_align)
                bptt = True

                # 3. Compute loss.
                try:
                    loss, batch_stats = self.train_loss(
                        batch,
                        outputs,
                        attns,
                        normalization=normalization,
                        shard_size=self.shard_size,
                        trunc_start=j,
                        trunc_size=trunc_size,
                    )

                    if loss is not None:
                        self.optim.backward(loss)

                    total_stats.update(batch_stats)
                    report_stats.update(batch_stats)

                except Exception:
                    traceback.print_exc()
                    logger.info("At step %d, we removed a batch - accum %d", self.optim.training_step, k)

                # 4. Update the parameters and statistics.
                if self.accum_count == 1:
                    # Multi GPU gradient gather
                    if self.n_gpu > 1:
                        grads = [p.grad.data for p in self.model.parameters() if p.requires_grad and p.grad is not None]
                        onmt.utils.distributed.all_reduce_and_rescale_tensors(grads, float(1))
                    self.optim.step()

                # If truncated, don't backprop fully.
                # TO CHECK
                # if dec_state is not None:
                #    dec_state.detach()
                if self.model.decoder.state is not None:
                    self.model.decoder.detach_state()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters() if p.requires_grad and p.grad is not None]
                onmt.utils.distributed.all_reduce_and_rescale_tensors(grads, float(1))
            self.optim.step()


class SimStateScoreTrainer(Trainer):
    def __init__(
        self,
        argmax_model,
        model,
        argmax_translator,
        translator,
        argmax_translation_builder,
        translation_builder,
        valid_translator,
        valid_builder,
        train_loss,
        valid_loss,
        optim,
        tgt_vocab,
        tgt_padding_token,
        reward_function,
        score_fn=None,
        trunc_size=0,
        shard_size=32,
        norm_method="sents",
        accum_count=[1],
        accum_steps=[0],
        n_gpu=1,
        gpu_rank=1,
        gpu_verbose_level=0,
        report_manager=None,
        with_align=False,
        model_saver=None,
        average_decay=0,
        average_every=1,
        model_dtype="fp32",
        earlystopper=None,
        dropout=[0.3],
        dropout_steps=[0],
        source_noise=None,
        metrics=None,
    ):
        super().__init__(
            model,
            train_loss,
            valid_loss,
            optim,
            trunc_size,
            shard_size,
            norm_method,
            accum_count,
            accum_steps,
            n_gpu,
            gpu_rank,
            gpu_verbose_level,
            report_manager,
            with_align,
            model_saver,
            average_decay,
            average_every,
            model_dtype,
            earlystopper,
            dropout,
            dropout_steps,
            source_noise,
        )
        self.avg_score = 0
        self.reward_function = reward_function
        self.metrics = [] if metrics is None else metrics
        self.translator = translator
        self.translation_builder = translation_builder
        self.valid_translator = valid_translator
        self.valid_builder = valid_builder
        self.score_fn = score_fn
        self.argmax_model = argmax_model
        self.argmax_translator = argmax_translator
        self.argmax_translation_builder = argmax_translation_builder
        self.tgt_vocab = tgt_vocab
        self.tgt_padding = tgt_padding_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def train(
        self,
        train_iter,
        train_steps,
        src_vocab=None,
        save_checkpoint_steps=5000,
        valid_iter=None,
        valid_steps=10000,
        stats_cls=None,
    ):
        """
        The main training loop by iterating over `train_iter` and possibly
        running validation on `valid_iter`.

        Args:
            train_iter: A generator that returns the next training batch.
            train_steps: Run training for this many iterations.
            save_checkpoint_steps: Save a checkpoint every this many
              iterations.
            valid_iter: A generator that returns the next validation batch.
            valid_steps: Run evaluation every this many iterations.

        Returns:
            The gathered statistics.
        """
        step = 0

        if not self.metrics:
            logger.info("No specific success metric mentioned")

        if valid_iter is None:
            logger.info("Start training loop without validation...")
        else:
            logger.info("Start training loop and validate every %d steps...", valid_steps)
        if stats_cls is None:
            total_stats = onmt.utils.Statistics()
            report_stats = onmt.utils.Statistics()
        else:
            total_stats = stats_cls()
            report_stats = stats_cls()
        self._start_report_manager(start_time=total_stats.start_time)
        all_metrics = {}
        self.validate(valid_iter, src_vocab, moving_average=self.moving_average, stats_cls=stats_cls)
        src_action_adv = []

        for batch in train_iter:
            src, src_lengths = batch.src if isinstance(batch.src, tuple) else (batch.src, None)
            if src_lengths is not None:
                report_stats.n_src_words += src_lengths.sum().item()

            with torch.no_grad():
                # get the sim score for the argmax as a baseline for the sample
                trans_batch = self.argmax_translator.translate_batch(
                    batch=batch, src_vocabs=[src_vocab], attn_debug=False
                )
                argmax_translations = self.argmax_translation_builder.from_batch(trans_batch)
                argmax_sim_values = [
                    self.reward_function(translation.pred_sents[0], translation.gold_sent)
                    for translation in argmax_translations
                ]

                # get the beam search top k and calculate sim_score for the actions
                trans_batch = self.translator.translate_batch(batch, [src_vocab], False)
                translations = self.translation_builder.from_batch(trans_batch)
                for i, translations_for_topk in enumerate(translations):
                    ref_code = translations_for_topk.gold_sent
                    src_raw = translations_for_topk.src_raw
                    argmax_reward = argmax_sim_values[i]
                    rewards = [
                        (pred_translation, float(self.reward_function(ref_code, pred_translation)))
                        for pred_translation in translations_for_topk.pred_sents
                    ]
                    top_trans = sorted(rewards, key=lambda x: x[1], reverse=True)
                    for pred_translation, reward in top_trans[:4]:
                        if reward < min(self.avg_score + 0.02, 1.0) or reward - argmax_reward < 0:
                            # print(f"reward is {reward} skip")
                            continue
                        actions_tensor = torch.tensor(
                            [2] + [self.tgt_vocab.stoi[token] for token in pred_translation] + [1]
                        )
                        src_tensor = torch.tensor([src_vocab.stoi[token] for token in src_raw])
                        src_action_adv.append((src_tensor, actions_tensor, reward))
                # sort by src length
            if len(src_action_adv) < 32:
                continue
            random.shuffle(src_action_adv)
            src_action_adv = src_action_adv[:32]
            src_action_adv = sorted(src_action_adv, key=lambda x: x[0].shape[0], reverse=True)
            src_data = [item[0] for item in src_action_adv]
            tgt_actions = [item[1] for item in src_action_adv]
            advs = torch.FloatTensor([item[2] for item in src_action_adv]).to(self.device)
            with torch.set_grad_enabled(True):
                try:
                    self.model.train()
                    self.optim.zero_grad()
                    src_lengths_eps = torch.tensor([src_tensor.shape[0] for src_tensor in src_data]).to(self.device)
                    batch_size = src_lengths_eps.shape[0]
                    src_eps = pad_sequence(src_data, padding_value=1, batch_first=True).to(self.device)
                    tgt_eps = pad_sequence(tgt_actions, padding_value=1, batch_first=True).to(self.device)
                    self.optim.zero_grad()

                    outputs, attns = self.model(
                        src_eps.permute(1, 0).view(-1, batch_size, 1),
                        tgt_eps.permute(1, 0).view(-1, batch_size, 1),
                        src_lengths_eps,
                        bptt=False,
                        with_align=self.with_align,
                    )
                    h_size = outputs.size(2)
                    bottled_output = outputs.permute(1, 0, 2).reshape(-1, h_size)
                    probs = self.model.generator(bottled_output)
                    # output_size = probs.size(-1)
                    indecies = tgt_eps
                    indecies = indecies[:, 1:]
                    log_prob = (
                        -F.nll_loss(probs, indecies.reshape(-1), ignore_index=1, reduction="none")
                        .view(batch_size, -1)
                        .mean(dim=-1)
                    )
                    reward = advs.reshape(-1, 1)
                    loss = (log_prob * -reward).mean()
                    loss.backward()
                    self.optim.step()
                except Exception as e:
                    print(e)
                    continue
            src_action_adv = []
            step += 1
            if step % 2 == 0:
                logger.info(f"Training step {step}")

            if valid_iter is not None and step % valid_steps == 0:
                self.model.eval()
                valid_stats = self.validate(
                    valid_iter, src_vocab, moving_average=self.moving_average, stats_cls=stats_cls
                )
            if train_steps > 0 and step >= train_steps:
                break

        if self.model_saver is not None:
            self.model_saver.save(step, moving_average=self.moving_average)
        return total_stats, all_metrics

    def validate(self, valid_iter, src_vocab, moving_average=None, stats_cls=None):
        """Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        valid_model = self.model
        if moving_average:
            # swap model params w/ moving average
            # (and keep the original parameters)
            model_params_data = []
            for avg, param in zip(self.moving_average, valid_model.parameters()):
                model_params_data.append(param.data)
                param.data = avg.data.half() if self.optim._fp16 == "legacy" else avg.data

        # Set model in validating mode.
        valid_model.eval()

        with torch.no_grad():
            if stats_cls is None:
                stats = onmt.utils.Statistics()
            else:
                stats = stats_cls()
            # reset metric
            for metric in self.metrics:
                metric.reset()

            for batch in tqdm(valid_iter):
                src, src_lengths = batch.src if isinstance(batch.src, tuple) else (batch.src, None)
                tgt = batch.tgt

                # F-prop through the model.
                outputs, attns = valid_model(src, tgt, src_lengths, with_align=self.with_align)

                # Compute loss.
                _, batch_stats = self.valid_loss(batch, outputs, attns)

                trans_batch = self.valid_translator.translate_batch(
                    batch=batch, src_vocabs=[src_vocab], attn_debug=False
                )
                translations = self.valid_builder.from_batch(trans_batch)
                targets = []
                preds = []
                avg_reward = 0
                counter = 0
                for trans in translations:
                    max_score = 0
                    pred_sent = trans.pred_sents[0]
                    reward = self.reward_function(trans.gold_sent, pred_sent)
                    if reward >= 0:
                        avg_reward += reward
                        counter += 1
                    preds.append(pred_sent)
                    targets.append(trans.gold_sent)
                avg_reward = avg_reward / counter if counter != 0 else -1000
                self.avg_score = avg_reward
                for metric in self.metrics:
                    metric.update(preds, targets)

                # Update statistics.
                stats.update(batch_stats)
            # if avg_score > self.avg_score:
            metrics_txt = ",".join(f"{metric}" for metric in self.metrics)
            logger.info(f"Validation metrics: {metrics_txt}")
        if moving_average:
            for param_data, param in zip(model_params_data, self.model.parameters()):
                param.data = param_data

        # Set model back to training mode.
        valid_model.train()

    def test(self, test_iter, src_vocab, stats_cls=None):
        """testing model.
                    test_iter: validate data iterator
                Returns:
                    :obj:`nmt.Statistics`: validation loss statistics
                """
        moving_average = self.moving_average
        test_model = self.model
        if moving_average:
            # swap model params w/ moving average
            # (and keep the original parameters)
            model_params_data = []
            for avg, param in zip(self.moving_average, test_model.parameters()):
                model_params_data.append(param.data)
                param.data = avg.data.half() if self.optim._fp16 == "legacy" else avg.data

        # Set model in validating mode.
        test_model.eval() #TODO : check if this function changes anything in the model

        with torch.no_grad():
            # print("at least we are in")
            if stats_cls is None:
                stats = onmt.utils.Statistics()
            else:
                stats = stats_cls()
            # reset metric
            for metric in self.metrics:
                metric.reset()

            for batch in tqdm(test_iter):
                src, src_lengths = batch.src if isinstance(batch.src, tuple) else (batch.src, None)
                tgt = batch.tgt

                # F-prop through the model.
                start = time.time()
                outputs, attns = test_model(src, tgt, src_lengths, with_align=self.with_align)
                end = time.time()
                print("this line took "+str(end - start))
                # Compute loss.
                _, batch_stats = self.valid_loss(batch, outputs, attns)

                trans_batch = self.valid_translator.translate_batch(
                    batch=batch, src_vocabs=[src_vocab], attn_debug=False
                )
                translations = self.valid_builder.from_batch(trans_batch)
                targets = []
                preds = []
                avg_reward = 0
                counter = 0
                for trans in translations:
                    max_score = 0
                    pred_sent = trans.pred_sents[0]
                    reward = self.reward_function(trans.gold_sent, pred_sent)
                    if reward >= 0:
                        avg_reward += reward
                        counter += 1
                    preds.append(pred_sent)
                    targets.append(trans.gold_sent)
                avg_reward = avg_reward / counter if counter != 0 else -1000
                self.avg_score = avg_reward
                for metric in self.metrics:
                    metric.update(preds, targets)

                # Update statistics.
                stats.update(batch_stats)
            # if avg_score > self.avg_score:
            metrics_txt = ",".join(f"{metric}" for metric in self.metrics)
            logger.info(f"Validation metrics: {metrics_txt}")
        if moving_average:
            for param_data, param in zip(model_params_data, self.model.parameters()):
                param.data = param_data

        # Set model back to training mode.
        test_model.train()

class SimStateScoreTrainerV2(Trainer):
    def __init__(
        self,
        argmax_model,
        model,
        argmax_translator,
        translator,
        argmax_translation_builder,
        translation_builder,
        valid_translator,
        valid_builder,
        train_loss,
        valid_loss,
        optim,
        tgt_vocab,
        tgt_padding_token,
        score_fn=None,
        trunc_size=0,
        shard_size=32,
        norm_method="sents",
        accum_count=[1],
        accum_steps=[0],
        n_gpu=1,
        gpu_rank=1,
        gpu_verbose_level=0,
        report_manager=None,
        with_align=False,
        model_saver=None,
        average_decay=0,
        average_every=1,
        model_dtype="fp32",
        earlystopper=None,
        dropout=[0.3],
        dropout_steps=[0],
        source_noise=None,
        metrics=None,
    ):
        super().__init__(
            model,
            train_loss,
            valid_loss,
            optim,
            trunc_size,
            shard_size,
            norm_method,
            accum_count,
            accum_steps,
            n_gpu,
            gpu_rank,
            gpu_verbose_level,
            report_manager,
            with_align,
            model_saver,
            average_decay,
            average_every,
            model_dtype,
            earlystopper,
            dropout,
            dropout_steps,
            source_noise,
        )
        self.metrics = [] if metrics is None else metrics
        self.translator = translator
        self.translation_builder = translation_builder
        self.valid_translator = valid_translator
        self.valid_builder = valid_builder
        self.score_fn = score_fn
        self.argmax_model = argmax_model
        self.argmax_translator = argmax_translator
        self.argmax_translation_builder = argmax_translation_builder
        self.tgt_vocab = tgt_vocab
        self.tgt_padding = tgt_padding_token

    def train(
        self,
        train_iter,
        train_steps,
        src_vocab=None,
        save_checkpoint_steps=5000,
        valid_iter=None,
        valid_steps=10000,
        stats_cls=None,
    ):
        """
        The main training loop by iterating over `train_iter` and possibly
        running validation on `valid_iter`.

        Args:
            train_iter: A generator that returns the next training batch.
            train_steps: Run training for this many iterations.
            save_checkpoint_steps: Save a checkpoint every this many
              iterations.
            valid_iter: A generator that returns the next validation batch.
            valid_steps: Run evaluation every this many iterations.

        Returns:
            The gathered statistics.
        """
        step = 0

        if not self.metrics:
            logger.info("No specific success metric mentioned")

        if valid_iter is None:
            logger.info("Start training loop without validation...")
        else:
            logger.info("Start training loop and validate every %d steps...", valid_steps)
        if stats_cls is None:
            total_stats = onmt.utils.Statistics()
            report_stats = onmt.utils.Statistics()
        else:
            total_stats = stats_cls()
            report_stats = stats_cls()
        self._start_report_manager(start_time=total_stats.start_time)
        all_metrics = {}
        # self.validate(
        #   valid_iter, src_vocab, moving_average=self.moving_average, stats_cls=stats_cls)
        src_action_adv = []

        for batch in train_iter:
            src, src_lengths = batch.src if isinstance(batch.src, tuple) else (batch.src, None)
            if src_lengths is not None:
                report_stats.n_src_words += src_lengths.sum().item()

            tgt_outer = batch.tgt

            with torch.no_grad():

                # get the beam search top k and calculate sim_score for the actions
                trans_batch = self.translator.translate_batch(batch=batch, src_vocabs=[src_vocab], attn_debug=False)
                translations = self.translation_builder.from_batch(trans_batch)
                for i, translations_for_topk in enumerate(translations):
                    ref_code = translations_for_topk.gold_sent
                    src_raw = translations_for_topk.src_raw
                    # argmax_reward = argmax_sim_values[i]
                    # if argmax_reward > 0.98: continue
                    pos_counter = 0
                    rewards_pred = [
                        (pred_translation, state_transition_reward(ref_code, pred_translation))
                        for pred_translation in translations_for_topk.pred_sents
                    ]
                    rewards_pred = sorted(rewards_pred, key=lambda x: sum(x[1]), reverse=True)

                    for pred_translation, rewards in rewards_pred[:3]:
                        rewards = checkpoints_reward(ref_code, pred_translation)
                        actions_tensor = torch.tensor(
                            [2] + [self.tgt_vocab.stoi[token] for token in pred_translation] + [3]
                        )
                        src_tensor = torch.tensor([src_vocab.stoi[token] for token in src_raw])
                        src_action_adv.append((src_tensor, actions_tensor, rewards + [0]))
                # sort by src length
            if len(src_action_adv) < 1:
                continue
            random.shuffle(src_action_adv)
            src_action_adv = src_action_adv[:128]
            src_action_adv = sorted(src_action_adv, key=lambda x: x[0].shape[0], reverse=True)
            src_data = [item[0] for item in src_action_adv]
            tgt_actions = [item[1] for item in src_action_adv]
            rewards = [item[2] for item in src_action_adv]
            with torch.set_grad_enabled(True):
                try:
                    self.model.train()
                    self.optim.zero_grad()
                    src_lengths_eps = torch.tensor([src_tensor.shape[0] for src_tensor in src_data]).to("cuda")
                    batch_size = src_lengths_eps.shape[0]
                    src_eps = pad_sequence(src_data, padding_value=1, batch_first=True).to("cuda")
                    tgt_eps = pad_sequence(tgt_actions, padding_value=1, batch_first=True).to("cuda")
                    padded_size = tgt_eps.size(-1)
                    rewards_padding = []
                    for rewards_token in rewards:
                        extra_padding = [0.0] * (padded_size - len(rewards_token) - 1)
                        rewards_padding.append(torch.FloatTensor(rewards_token + extra_padding))

                    self.optim.zero_grad()

                    outputs, attns = self.model(
                        src_eps.permute(1, 0).view(-1, batch_size, 1),
                        tgt_eps.permute(1, 0).view(-1, batch_size, 1),
                        src_lengths_eps,
                        bptt=False,
                        with_align=self.with_align,
                    )
                    h_size = outputs.size(2)
                    bottled_output = outputs.permute(1, 0, 2).reshape(-1, h_size)
                    log_prob_v = self.model.generator(bottled_output)
                    indecies = tgt_eps
                    indecies = indecies[:, 1:]
                    indecies = indecies.reshape(-1)
                    adv_v = torch.cat(rewards_padding).reshape(-1, 1).to("cuda")
                    mask = torch.zeros_like(log_prob_v)
                    mask[indecies == 1, :] = 1.0
                    log_prob_v = log_prob_v.masked_fill(mask.bool(), 0.0)
                    lp_a = log_prob_v[range(indecies.size(0)), indecies]
                    log_prob_actions_v = adv_v * lp_a
                    loss_policy_v = -log_prob_actions_v.mean()
                    loss_v = loss_policy_v
                    loss_v.backward()
                    self.optim.step()
                except Exception as e:
                    print(e)
                    continue
            src_action_adv = []
            step += 1
            logger.info(f"Training step {step}")

            if valid_iter is not None and step % valid_steps == 0:
                self.model.eval()
                valid_stats = self.validate(
                    valid_iter, src_vocab, moving_average=self.moving_average, stats_cls=stats_cls
                )

        if self.model_saver is not None:
            self.model_saver.save(step, moving_average=self.moving_average)
        return total_stats, all_metrics

    def validate(self, valid_iter, src_vocab, moving_average=None, stats_cls=None):
        """Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        valid_model = self.model
        if moving_average:
            # swap model params w/ moving average
            # (and keep the original parameters)
            model_params_data = []
            for avg, param in zip(self.moving_average, valid_model.parameters()):
                model_params_data.append(param.data)
                param.data = avg.data.half() if self.optim._fp16 == "legacy" else avg.data

        # Set model in validating mode.
        valid_model.eval()

        with torch.no_grad():
            if stats_cls is None:
                stats = onmt.utils.Statistics()
            else:
                stats = stats_cls()
            # reset metric
            for metric in self.metrics:
                metric.reset()

            for batch in valid_iter:
                src, src_lengths = batch.src if isinstance(batch.src, tuple) else (batch.src, None)
                tgt = batch.tgt

                # F-prop through the model.
                outputs, attns = valid_model(src, tgt, src_lengths, with_align=self.with_align)

                # Compute loss.
                _, batch_stats = self.valid_loss(batch, outputs, attns)

                trans_batch = self.valid_translator.translate_batch(
                    batch=batch, src_vocabs=[src_vocab], attn_debug=False
                )
                translations = self.valid_builder.from_batch(trans_batch)
                targets = []
                preds = []
                for trans in translations:
                    max_score = 0
                    pred_sent = trans.pred_sents[0]
                    if self.score_fn is not None:
                        for pred_seq in trans.pred_sents:
                            self.score_fn.reset()
                            self.score_fn.update([pred_seq], [trans.gold_sent])
                            score = self.score_fn.eval()
                            if score > max_score:
                                pred_sent = pred_seq
                                max_score = score
                    preds.append(pred_sent)
                    targets.append(trans.gold_sent)

                for metric in self.metrics:
                    metric.update(preds, targets)

                # Update statistics.
                stats.update(batch_stats)
            metrics_txt = ",".join(f"{metric}" for metric in self.metrics)
            logger.info(f"Validation metrics: {metrics_txt}")
        if moving_average:
            for param_data, param in zip(model_params_data, self.model.parameters()):
                param.data = param_data

        # Set model back to training mode.
        valid_model.train()


class SimCheckpointRewardTrainer(Trainer):
    def __init__(
        self,
        argmax_model,
        model,
        argmax_translator,
        translator,
        argmax_translation_builder,
        translation_builder,
        valid_translator,
        valid_builder,
        train_loss,
        valid_loss,
        optim,
        tgt_vocab,
        tgt_padding_token,
        score_fn=None,
        trunc_size=0,
        shard_size=32,
        norm_method="sents",
        accum_count=[1],
        accum_steps=[0],
        n_gpu=1,
        gpu_rank=1,
        gpu_verbose_level=0,
        report_manager=None,
        with_align=False,
        model_saver=None,
        average_decay=0,
        average_every=1,
        model_dtype="fp32",
        earlystopper=None,
        dropout=[0.3],
        dropout_steps=[0],
        source_noise=None,
        metrics=None,
    ):
        super().__init__(
            model,
            train_loss,
            valid_loss,
            optim,
            trunc_size,
            shard_size,
            norm_method,
            accum_count,
            accum_steps,
            n_gpu,
            gpu_rank,
            gpu_verbose_level,
            report_manager,
            with_align,
            model_saver,
            average_decay,
            average_every,
            model_dtype,
            earlystopper,
            dropout,
            dropout_steps,
            source_noise,
        )
        self.metrics = [] if metrics is None else metrics
        self.translator = translator
        self.translation_builder = translation_builder
        self.valid_translator = valid_translator
        self.valid_builder = valid_builder
        self.score_fn = score_fn
        self.argmax_model = argmax_model
        self.argmax_translator = argmax_translator
        self.argmax_translation_builder = argmax_translation_builder
        self.tgt_vocab = tgt_vocab
        self.tgt_padding = tgt_padding_token
        self.simcode_tokenizer = SimCodeTokenizer()

    def train(
        self,
        train_iter,
        train_steps,
        src_vocab=None,
        save_checkpoint_steps=5000,
        valid_iter=None,
        valid_steps=10000,
        stats_cls=None,
    ):
        """
        The main training loop by iterating over `train_iter` and possibly
        running validation on `valid_iter`.

        Args:
            train_iter: A generator that returns the next training batch.
            train_steps: Run training for this many iterations.
            save_checkpoint_steps: Save a checkpoint every this many
              iterations.
            valid_iter: A generator that returns the next validation batch.
            valid_steps: Run evaluation every this many iterations.

        Returns:
            The gathered statistics.
        """
        step = 0

        if not self.metrics:
            logger.info("No specific success metric mentioned")

        if valid_iter is None:
            logger.info("Start training loop without validation...")
        else:
            logger.info("Start training loop and validate every %d steps...", valid_steps)
        if stats_cls is None:
            total_stats = onmt.utils.Statistics()
            report_stats = onmt.utils.Statistics()
        else:
            total_stats = stats_cls()
            report_stats = stats_cls()
        self._start_report_manager(start_time=total_stats.start_time)
        all_metrics = {}
        self.validate(valid_iter, src_vocab, moving_average=self.moving_average, stats_cls=stats_cls)
        src_action_adv = []

        for i, (batches, normalization) in enumerate(self._accum_batches(train_iter)):
            for k, batch in enumerate(batches):
                # print("Start to get data")
                # get the beam search top k and calculate sim_score for the actions
                trans_batch = self.argmax_translator.translate_batch(
                    batch=batch, src_vocabs=[src_vocab], attn_debug=False
                )
                translations = self.argmax_translation_builder.from_batch(trans_batch)
                argmax_sim_values = [
                    sum(checkpoints_reward(translation.gold_sent, translation.pred_sents[0]))
                    for translation in translations
                ]

                # get the beam search top k and calculate sim_score for the actions

                trans_batch = self.translator.translate_batch(batch=batch, src_vocabs=[src_vocab], attn_debug=False)
                translations = self.translation_builder.from_batch(trans_batch)
                for i, translations_for_topk in enumerate(translations):
                    ref_code = translations_for_topk.gold_sent
                    src_raw = translations_for_topk.src_raw
                    argmax_reward = argmax_sim_values[i]
                    elements = []
                    for pred_translation in translations_for_topk.pred_sents:
                        reward = checkpoints_reward(ref_code, pred_translation)
                        if sum(reward) <= argmax_reward + 0.01:
                            continue
                        advantages = torch.tensor([0.0] + reward)
                        actions_tensor = torch.tensor(
                            [2] + [self.tgt_vocab.stoi[token] for token in pred_translation] + [3]
                        )
                        if advantages.shape[0] + 1 != actions_tensor.shape[0]:
                            print("length missmatch")
                            continue
                        src_tensor = torch.tensor([src_vocab.stoi[token] for token in src_raw])
                        elements.append((sum(reward), (src_tensor, actions_tensor, advantages)))
                    elements = sorted(elements, key=lambda x: x[0], reverse=True)
                    for _, eps in elements[:8]:
                        src_action_adv.append(eps)
                    if len(elements) > 0:
                        print(len(src_action_adv))

                    # sort by src length
            if len(src_action_adv) < 64:
                continue
            left_overs = []  # src_action_adv[64:]
            src_action_adv = src_action_adv[:64]
            src_action_adv = sorted(src_action_adv, key=lambda x: x[0].shape[0], reverse=True)
            src_data = [item[0] for item in src_action_adv]
            tgt_actions = [item[1] for item in src_action_adv]
            advs = [item[2] for item in src_action_adv]
            # print("Got data apply model")
            with torch.set_grad_enabled(True):
                try:
                    self.model.train()
                    self.optim.zero_grad()
                    src_lengths_eps = torch.tensor([src_tensor.shape[0] for src_tensor in src_data]).to("cuda")
                    src_eps = pad_sequence(src_data, padding_value=1).view(-1, 64, 1).to("cuda")
                    tgt_eps = pad_sequence(tgt_actions, padding_value=1).view(-1, 64, 1).to("cuda")
                    # calculate the actions to target output
                    outputs, attns = self.model(src_eps, tgt_eps, src_lengths_eps, with_align=False)
                    policy_prob = self.model.generator(outputs).permute(1, 0, 2)

                    relevent_policy = torch.cat(
                        [policy_prob[idx][: len(actions) - 1] for idx, actions in enumerate(tgt_actions)]
                    )
                    actions = [actions[1:] for actions in tgt_actions]
                    actions_v = torch.cat(actions)
                    actions_l = actions_v.tolist()
                    actions_v.to("cuda")
                    advs_v = torch.cat(advs).to("cuda")
                    log_prob_v = F.log_softmax(relevent_policy, dim=1)
                    lp_a = log_prob_v[range(len(actions_l)), actions_v]
                    log_prob_actions_v = advs_v * lp_a
                    loss_policy_v = -log_prob_actions_v.mean()
                    loss_v = loss_policy_v
                    loss_v.backward()
                    self.optim.step()
                    src_action_adv = left_overs
                except Exception as e:
                    print(e)
                    continue
            # print("Finished model ")
            step += 1
            if step % 2 == 0:
                logger.info(f"Training step {step}")
            if valid_iter is not None and step % valid_steps == 0:
                self.model.eval()
                if self.gpu_verbose_level > 0:
                    logger.info("GpuRank %d: validate step %d" % (self.gpu_rank, step))
                valid_stats = self.validate(
                    valid_iter, src_vocab, moving_average=self.moving_average, stats_cls=stats_cls
                )
                all_metrics[str(i)] = []
                for metric in self.metrics:
                    all_metrics[str(i)].append(metric.eval())
                if self.gpu_verbose_level > 0:
                    logger.info(
                        "GpuRank %d: gather valid stat \
                                        step %d"
                        % (self.gpu_rank, step)
                    )
                valid_stats = self._maybe_gather_stats(valid_stats)
                if self.gpu_verbose_level > 0:
                    logger.info("GpuRank %d: report stat step %d" % (self.gpu_rank, step))
                self._report_step(self.optim.learning_rate(), step, valid_stats=valid_stats)
                # Run patience mechanism
                if self.earlystopper is not None:
                    self.earlystopper(valid_stats, step)
                    # If the patience has reached the limit, stop training
                    if self.earlystopper.has_stopped():
                        break

            if self.model_saver is not None and (save_checkpoint_steps != 0 and step % save_checkpoint_steps == 0):
                self.model_saver.save(step, moving_average=self.moving_average)

            if train_steps > 0 and step >= train_steps:
                break

        if self.model_saver is not None:
            self.model_saver.save(step, moving_average=self.moving_average)
        return total_stats, all_metrics

    def validate(self, valid_iter, src_vocab, moving_average=None, stats_cls=None):
        """Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        valid_model = self.model
        if moving_average:
            # swap model params w/ moving average
            # (and keep the original parameters)
            model_params_data = []
            for avg, param in zip(self.moving_average, valid_model.parameters()):
                model_params_data.append(param.data)
                param.data = avg.data.half() if self.optim._fp16 == "legacy" else avg.data

        # Set model in validating mode.
        valid_model.eval()

        with torch.no_grad():
            if stats_cls is None:
                stats = onmt.utils.Statistics()
            else:
                stats = stats_cls()
            # reset metric
            for metric in self.metrics:
                metric.reset()

            for batch in valid_iter:
                src, src_lengths = batch.src if isinstance(batch.src, tuple) else (batch.src, None)
                tgt = batch.tgt

                # F-prop through the model.
                outputs, attns = valid_model(src, tgt, src_lengths, with_align=self.with_align)

                # Compute loss.
                _, batch_stats = self.valid_loss(batch, outputs, attns)

                trans_batch = self.valid_translator.translate_batch(
                    batch=batch, src_vocabs=[src_vocab], attn_debug=False
                )
                translations = self.valid_builder.from_batch(trans_batch)
                targets = []
                preds = []
                for trans in translations:
                    max_score = 0
                    pred_sent = trans.pred_sents[0]
                    if self.score_fn is not None:
                        for pred_seq in trans.pred_sents:
                            self.score_fn.reset()
                            self.score_fn.update([pred_seq], [trans.gold_sent])
                            score = self.score_fn.eval()
                            if score > max_score:
                                pred_sent = pred_seq
                                max_score = score
                    preds.append(pred_sent)
                    targets.append(trans.gold_sent)

                for metric in self.metrics:
                    metric.update(preds, targets)

                # Update statistics.
                stats.update(batch_stats)
            metrics_txt = ",".join(f"{metric}" for metric in self.metrics)
            logger.info(f"Validation metrics: {metrics_txt}")
        if moving_average:
            for param_data, param in zip(model_params_data, self.model.parameters()):
                param.data = param_data

        # Set model back to training mode.
        valid_model.train()
