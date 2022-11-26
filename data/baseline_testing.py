import json
import argparse
import random

import onmt


# import torch

# from grammer.utils import convert_tokens_to_code, execute_simcode


class RobertasAnswer:
    def __init__(
            self,
            model,
            valid_translator,
            valid_builder
    ):
        self.valid_builder = valid_builder
        self.valid_translator = valid_translator
        self.model = model

    def predict(self, valid_iter, src_vocab):
        """
        return the roberta answer for the given question
        """
        valid_model = self.model
        returned_answers = []
        # Set model in validating mode.
        valid_model.eval()

        with torch.no_grad():
            # reset metric
            for batch in valid_iter:
                trans_batch = self.valid_translator.translate_batch(
                    batch=batch, src_vocabs=[src_vocab], attn_debug=False
                )
                translations = self.valid_builder.from_batch(trans_batch)
                targets = []
                preds = []
                for trans in translations:
                    pred_sent = trans.pred_sents[0]
                    preds.append(pred_sent)
                    targets.append(trans.gold_sent)

                    for pred_seq, tgt_seq in zip(preds, targets):
                        pred_code = convert_tokens_to_code(pred_seq)
                        try:
                            pred_answer, pred_state = execute_simcode(pred_code, True)
                            returned_answers.append(pred_answer)
                        except Exception as e:
                            pass
            return returned_answers


def main():
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--train_percent', type=int, help='which percent of the train json multiple choice answers'
                                                          ' you would like to change to yes/no questions')
    parser.add_argument('--test_percent', type=int, help='which percent of the test json multiple choice answers'
                                                         ' you would like to change to yes/no questions')
    parser.add_argument('--version', type=str, help='v1 or v2, to which volume  ')
    args = parser.parse_args()

    f = open('data/' + args.version + '/original_train.json', 'r')
    src_train = json.loads(f.read())
    with open('y_n_multiple_train.json', 'w') as added_file:
        to_insert = {}
        for sec in src_train:
            if sec == 'data':
                entries_to_be_changed = random.sample(range(0, len(src_train['data'])),
                                                      len(src_train['data']) * args.train_percent // 100)
                print(entries_to_be_changed)
                to_insert['data'] = []
                for entry in range(len(src_train['data'])):
                    if entry in entries_to_be_changed:
                        coin_toss = random.choice([True, False])
                        correct_val_answer = src_train["data"][entry]["correct_answer"]
                        all_wrong_answers = src_train["data"][entry]["answers"]
                        all_wrong_answers.remove(correct_val_answer)
                        if coin_toss:
                            to_insert['data'].append({"context": src_train["data"][entry]["context"],
                                                      "question": src_train["data"][entry]["question"],
                                                      "correct_answer": correct_val_answer,
                                                      "answers": [correct_val_answer, "False", correct_val_answer,
                                                                  "False"],
                                                      "code": src_train["data"][entry]["code"]
                                                      })
                        else:
                            wrong = random.choice(all_wrong_answers)
                            to_insert['data'].append({"context": src_train["data"][entry]["context"],
                                                      "question": src_train["data"][entry]["question"],
                                                      "correct_answer": "False",
                                                      "answers": [wrong, "False", wrong, "False"],
                                                      "code": src_train["data"][entry]["code"]
                                                      })
                    else:
                        to_insert['data'].append(src_train['data'][entry])
            else:
                to_insert[sec] = src_train[sec]
        json.dump(to_insert, added_file)

    f = open('data/' + args.version + '/original_test.json', 'r')
    src_test = json.loads(f.read())
    with open('y_n_multiple_test.json', 'w') as added_test_file:
        for sec in src_test:
            if sec == 'data':
                entries_to_be_changed = random.sample(range(0, len(src_test['data'])),
                                                      len(src_test['data']) * args.test_percent // 100)
                print(entries_to_be_changed)
                to_insert['data'] = []
                for entry in range(len(src_test['data'])):
                    if entry in entries_to_be_changed:
                        coin_toss = random.choice([True, False])
                        correct_val_answer = src_test["data"][entry]["correct_answer"]
                        all_wrong_answers = src_test["data"][entry]["answers"]
                        if correct_val_answer in all_wrong_answers:
                            all_wrong_answers.remove(correct_val_answer)
                        if coin_toss:
                            to_insert['data'].append({"context": src_test["data"][entry]["context"],
                                                      "question": src_test["data"][entry]["question"],
                                                      "correct_answer": correct_val_answer,
                                                      "answers": [correct_val_answer, "False", correct_val_answer,
                                                                  "False"],
                                                      "code": src_test["data"][entry]["code"]
                                                      })
                        else:
                            wrong = random.choice(all_wrong_answers)
                            to_insert['data'].append({"context": src_test["data"][entry]["context"],
                                                      "question": src_test["data"][entry]["question"],
                                                      "correct_answer": "False",
                                                      "answers": [wrong, "False", wrong, "False"],
                                                      "code": src_test["data"][entry]["code"]
                                                      })
                    else:
                        to_insert['data'].append(src_test['data'][entry])
            else:
                to_insert[sec] = src_test[sec]
        json.dump(to_insert, added_test_file)


if __name__ == "__main__":
    main()

    # valid_data_file = f"data/v1/processed/SimQA.valid.0.pt"
    # vocab_data_file = f"data/v1/processed/SimQA.vocab.pt"
    # vocab_fields = torch.load(vocab_data_file)
    #
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    #
    # valid_iter = onmt.inputters.inputter.DatasetLazyIter(
    #     dataset_paths=[valid_data_file],
    #     fields=vocab_fields,
    #     batch_size=32,
    #     batch_size_multiple=1,
    #     batch_size_fn=None,
    #     device=device,
    #     is_train=False,
    #     repeat=False,
    #     pool_factor=8192,
    # )
    #
    # src_text_field = vocab_fields["src"].base_field
    # src_vocab = src_text_field.vocab
    # model_path = f"models/v1/src.embeddings.pt"
    # src_padding = src_vocab.stoi[src_text_field.pad_token]
    #
    # encoder_embeddings = onmt.modules.Embeddings(
    #     200, len(src_vocab), word_padding_idx=src_padding, position_encoding=True, fix_word_vecs=True
    # )
    # encoder = onmt.encoders.TransformerEncoder(
    #     num_layers=6,
    #     d_model=200,
    #     max_relative_positions=100,
    #     embeddings=encoder_embeddings,
    #     d_ff=1,
    #     dropout=0.1,
    #     attention_dropout=0.1,
    #     heads=8,
    # )
    #
    # tgt_text_field = vocab_fields["tgt"].base_field
    # tgt_vocab = tgt_text_field.vocab
    # tgt_padding = tgt_vocab.stoi[tgt_text_field.pad_token]
    # decoder_embeddings = onmt.modules.Embeddings(
    #     200, len(tgt_vocab), word_padding_idx=tgt_padding, position_encoding=True
    # )
    # decoder = onmt.decoders.TransformerDecoder(
    #     num_layers=6,
    #     d_model=200,
    #     embeddings=decoder_embeddings,
    #     max_relative_positions=100,
    #     aan_useffn=False,
    #     full_context_alignment=True,
    #     alignment_layer=1,
    #     alignment_heads=1,
    #     heads=8,
    #     d_ff=1,
    #     copy_attn=False,
    #     self_attn_type="scaled-dot",
    #     dropout=0.1,
    #     attention_dropout=0.1,
    # )
    # device_id = 0 if torch.cuda.is_available() else -1
    # model = onmt.models.model.NMTModel(encoder, decoder).to(device)
    #
    # vld_translator = onmt.translate.Translator(
    #     model=model,
    #     fields=vocab_fields,
    #     src_reader=onmt.inputters.str2reader["text"],
    #     tgt_reader=onmt.inputters.str2reader["text"],
    #     global_scorer=onmt.translate.GNMTGlobalScorer(
    #         alpha=0.7, beta=0.0, length_penalty="avg", coverage_penalty="none"
    #     ),
    #     gpu=device_id,
    # )
    # valid_builder = onmt.translate.TranslationBuilder(
    #     data=torch.load(valid_data_file), fields=vocab_fields, has_tgt=True
    # ),
    # answering_mode = RobertasAnswer(model=model_path, valid_translator=vld_translator, valid_builder=valid_builder)
    # print(answering_mode.predict(valid_iter, src_vocab))
