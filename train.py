import datetime
import json
import logging
import time

import tensorflow as tf
from pathlib2 import Path
from sklearn import model_selection

import data
import model
import vocab
from data import process_raw_data, build_dataset_op, Task
from flags import define_flags
from stc3dataset.data.eval import evaluate_from_list
from vocab import Language, HelpDeskType

PROJECT_DIR = Path(__file__).parent.parent
import numpy as np

def flags2params(flags, customized_params=None):
    if customized_params:
        flags.__dict__.update(customized_params)

    flags.checkpoint_dir = Path(flags.checkpoint_dir) / flags.language / flags.task

    flags.output_dir.mkdir(parents=True, exist_ok=True)

    flags.best_model_dir = Path(flags.best_model_dir) / flags.language / flags.task

    flags.language = vocab.Language[flags.language]
    flags.task = data.Task[flags.task]
    if flags.language == Language.english:
        flags.vocab = getattr(vocab, flags.english_vocab)
    else:
        flags.vocab = getattr(vocab, flags.chinese_vocab)

    flags.optimizer = getattr(tf.train, flags.optimizer)
    flags.cell = getattr(tf.nn.rnn_cell, flags.cell)

    flags.h_type = HelpDeskType[flags.h_type]

    return flags

class TrainingHelper(object):
    def __init__(self, customized_params=None, log_to_tensorboard=True):
        # Parse parameters
        flags = define_flags()
        params = flags2params(flags, customized_params)

        self.logger = logging.getLogger(__name__)
        self.logger.info("Task: " + str(params.task))
        self.logger.info("Language: " + str(params.language))
        self.task = params.task
        self.language = params.language
        self.h_type = params.h_type
        self.run_name = "%s_%s_%s_%s_%s" % (
            params.tag, self.task.name, self.language.name, self.h_type.name,
            datetime.datetime.now().strftime('%b-%d_%H-%M-%S-%f'))
        assert not (params.log_dir / self.run_name).is_dir(), "The run %s has existed in Log Path %s" % (
        self.run_name, params.log_dir)
        self.checkpoint_dir = params.checkpoint_dir / self.run_name / "model"
        self.best_model_dir = params.best_model_dir / self.run_name / "best"
        self.output_dir = params.output_dir


        # Loading dataset
        train_path, test_path, vocab = prepare_data_and_vocab(
            vocab=params.vocab,
            store_folder=params.embedding_dir,
            data_dir=params.data_dir,
            language=params.language,
            h_type=params.h_type,
        )

        # split training set into train and dev sets
        self.raw_train, self.raw_dev = model_selection.train_test_split(
            json.load(train_path.open(encoding='utf-8')),
            test_size=params.dev_ratio, random_state=params.random_seed)

        self.raw_test = json.load(test_path.open(encoding='utf-8'))

        train_dataset = process_raw_data(
            self.raw_train,
            vocab=vocab,
            max_len=params.max_len,
            cache_dir=params.cache_dir,
            is_train=True,
            name="train_%s_%s" % (params.language, params.h_type))

        train_eval_dataset = process_raw_data(
            self.raw_train,
            vocab=vocab,
            max_len=params.max_len,
            cache_dir=params.cache_dir,
            is_train=False,
            name="train_eval_%s_%s" % (params.language, params.h_type))

        dev_dataset = process_raw_data(
            self.raw_dev,
            vocab=vocab,
            max_len=params.max_len,
            cache_dir=params.cache_dir,
            is_train=False,
            name="dev_%s_%s" % (params.language, params.h_type))

        test_dataset = process_raw_data(
            self.raw_test,
            vocab=vocab,
            max_len=params.max_len,
            cache_dir=params.cache_dir,
            is_train=False,
            name="test_%s_%s" % (params.language, params.h_type))

        pad_idx = vocab.pad_idx
        self.train_iterator = build_dataset_op(train_dataset, pad_idx, params.batch_size, is_train=True)
        self.train_batch = self.train_iterator.get_next()

        self.train_eval_iterator = build_dataset_op(train_eval_dataset, pad_idx, params.batch_size, is_train=False)
        self.train_eval_batch = self.train_eval_iterator.get_next()

        self.dev_iterator = build_dataset_op(dev_dataset, pad_idx, params.batch_size, is_train=False)
        self.dev_batch = self.dev_iterator.get_next()
        self.test_iterator = build_dataset_op(test_dataset, pad_idx, params.batch_size, is_train=False)
        self.test_batch = self.test_iterator.get_next()

        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)

        self.model = model.Model(vocab.weight, self.task, params, session=sess)
        self.inference_mode = False
        self.num_epoch = params.num_epoch

        if params.resume_dir:
            self.model.load_model(params.resume_dir)
            if params.infer_test:
                self.inference_mode = True
            self.logger.info("Inference_mode: On")

        if log_to_tensorboard:
            self.log_to_tensorboard = log_to_tensorboard
            self.log_writer = tf.summary.FileWriter(
                str(params.log_dir / self.run_name),
                sess.graph,
                flush_secs=20)

    def train_epoch(self, checkpoint_dir=None):
        train_loss, c_shape, h_shape = self.model.train_epoch(
            self.train_iterator.initializer,
            self.train_batch,
            save_path=checkpoint_dir or self.checkpoint_dir)
        self.logger.info("C" + c_shape)
        self.logger.info("H" + h_shape)
        return train_loss

    def load_best_model(self):
        self.model.load_model(self.best_model_dir.parent.resolve())

    def train(self, num_epoch=None):
        best_score = 100
        for epoch in range(num_epoch or self.num_epoch):
            start = time.time()
            train_loss = self.train_epoch()
            used_time = time.time() - start
            self.logger.info(" Epoch %d, training loss = %.4f, used %.2f sec" % (epoch + 1, train_loss, used_time))
            train_metrics = self.evaluate_on_train()
            dev_metrics = self.evaluate_on_dev()
            curr_score = self.metrics_to_single_value(dev_metrics)

            self.logger.info(" Train Metrics: %s" % train_metrics[self.task.name])
            self.logger.info(" Dev Metrics: %s" % dev_metrics[self.task.name])

            if self.log_to_tensorboard:
                self.write_to_summary(train_metrics, dev_metrics, train_loss, epoch)

            if best_score > curr_score:
                best_score = curr_score
                self.model.save_model(self.best_model_dir)

        self.load_best_model()


    def write_to_summary(self, train_metrics, dev_metrics, train_loss, global_step):
        summary = tf.Summary()
        if dev_metrics["quality"] is not None:
            for distance_type, distance in dev_metrics["quality"].items():
                for score_type, score in distance.items():
                    summary.value.add(tag="quality_dev_%s/%s_score" % (distance_type, score_type), simple_value=score)

        summary.value.add(tag="train_loss", simple_value=train_loss)

        if train_metrics["nugget"] is not None:
            for distance_type, distance in train_metrics["nugget"].items():
                summary.value.add(tag="nugget_train_%s/" % (distance_type), simple_value=distance)

        if dev_metrics["nugget"] is not None:
            for distance_type, distance in dev_metrics["nugget"].items():
                summary.value.add(tag="nugget_dev_%s/" % (distance_type), simple_value=distance)

        self.log_writer.add_run_metadata(self.model.run_metadata, "meta_%s" % global_step, global_step=global_step)
        self.log_writer.add_summary(summary, global_step=global_step)

    def evaluate_on_train(self):
        predictions = self.model.predict(self.train_eval_iterator.initializer, self.train_eval_batch)
        submission = self.__predictions_to_submission_format(predictions)
        scores = evaluate_from_list(submission, self.raw_train)
        return scores

    def evaluate_on_dev(self):
        predictions = self.model.predict(self.dev_iterator.initializer, self.dev_batch)
        submission = self.__predictions_to_submission_format(predictions)
        scores = evaluate_from_list(submission, self.raw_dev)
        return scores

    def predict_test(self, write_to_file=True):
        predictions = self.model.predict(self.test_iterator.initializer, self.test_batch)
        submission = self.__predictions_to_submission_format(predictions)

        if write_to_file:
            output_file = trainer.output_dir / ("%s_%s_%s_test_submission.json" %
                                                (self.task.name, self.language.name, self.h_type.name))
            output_file.parent.mkdir(parents=True, exist_ok=True)
            json.dump(submission, output_file.open("w"))

        return submission

    def __predictions_to_submission_format(self, predictions):
        submission = []
        for pred in predictions:
            if self.task == Task.nugget:
                submission.append(data.nugget_prediction_to_submission_format(pred))
            elif self.task == Task.quality:
                submission.append(data.quality_prediction_to_submission_format(pred))
        return submission

    def metrics_to_single_value(self, metrics):
        if self.task == Task.nugget:
            return metrics["nugget"]["rnss"]
        if self.task == Task.quality:
            return np.mean(list(metrics["quality"]["rsnod"].values()))



def prepare_data_and_vocab(vocab, store_folder, data_dir, language=Language.english, h_type=None, tokenizer=None):
    tf.gfile.MakeDirs(str(store_folder))
    if language == Language.chinese:
        if h_type == HelpDeskType.CZ:
            train_path = data_dir / "train_data_cn_CZ.json"
            test_path = data_dir / "test_data_cn_CZ.json"
        elif h_type == HelpDeskType.DX:
            train_path = data_dir / "train_data_cn_DX.json"
            test_path = data_dir / "test_data_cn_DX.json"
        elif h_type == HelpDeskType.LT:
            train_path = data_dir / "train_data_cn_LT.json"
            test_path = data_dir / "test_data_cn_LT.json"
        elif h_type == HelpDeskType.OT:
            train_path = data_dir / "train_data_cn_OT.json"
            test_path = data_dir / "test_data_cn_OT.json"
        else:
            train_path = data_dir / "train_data_cn_ALL.json"
            test_path = data_dir / "test_data_cn_ALL.json"
    else:
        train_path = data_dir / "train_data_en.json"
        test_path = data_dir / "test_data_en.json"

    vocab = vocab(store_folder, tokenizer=tokenizer)
    return train_path, test_path, vocab


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    trainer = TrainingHelper()
    if not trainer.inference_mode:
        trainer.train()

    test_prediction = trainer.predict_test()
