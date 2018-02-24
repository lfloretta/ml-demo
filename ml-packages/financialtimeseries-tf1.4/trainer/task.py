import argparse
import os

import trainer.model as model

import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam


def run_experiment(hparams):
  """Run the training and evaluate using the high level API"""
  # TrainSpec
  train_spec = tf.estimator.TrainSpec(
      input_fn = lambda: model.csv_input_fn(
          hparams.train_files,
          mode=tf.estimator.ModeKeys.TRAIN,
          num_epochs= hparams.num_epochs,
          batch_size = hparams.train_batch_size
      ),
      max_steps=hparams.max_steps,
  )

  # EvalSpec
  eval_spec = tf.estimator.EvalSpec(
      input_fn =lambda: model.csv_input_fn(
          hparams.eval_files,
          batch_size = hparams.eval_batch_size
      ),
      exporters=[tf.estimator.LatestExporter(
          name="classifier",  # the name of the folder in which the model will be exported to under export
          serving_input_receiver_fn=model.json_serving_input_fn,
          exports_to_keep=1,
         as_text=True)],
      steps = None,
      throttle_secs = hparams.evaluate_after_sec # evalute after each 10 training seconds!
  )


  run_config = tf.estimator.RunConfig()
  run_config = run_config.replace(model_dir=hparams.job_dir)
  print('model dir {}'.format(run_config.model_dir))
  dnn_estimator = model.create_DNNLinearCombinedClassifier(run_config, hparams)

  tf.estimator.train_and_evaluate(
      dnn_estimator,
      train_spec,
      eval_spec
  )

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # Input Arguments
  parser.add_argument(
      '--train-files',
      help='GCS or local paths to training data',
      nargs='+',
      required=True
  )
  parser.add_argument(
      '--num-epochs',
      help="""\
      Maximum number of training data epochs on which to train.
      If both --max-steps and --num-epochs are specified,
      the training job will run for --max-steps or --num-epochs,
      whichever occurs first. If unspecified will run for --max-steps.\
      """,
      type=int,
  )
  parser.add_argument(
      '--max-steps',
      help="""\
      Maximum number of training steps on which to train.
      If both --max-steps and --num-epochs are specified,
      the training job will run for --max-steps or --num-epochs,
      whichever occurs first. If unspecified will run for --max-steps.\
      """,
      type=int,
  )
  parser.add_argument(
      '--train-batch-size',
      help='Batch size for training steps',
      type=int,
      default=40
  )
  parser.add_argument(
      '--eval-batch-size',
      help='Batch size for evaluation steps',
      type=int,
      default=40
  )
  parser.add_argument(
      '--eval-files',
      help='GCS or local paths to evaluation data',
      nargs='+',
      required=True
  )
  # Training arguments
  parser.add_argument(
      '--learning-rate',
      help='Learning rate',
      default=0.01,
      type=float
  )
  parser.add_argument(
      '--first-layer-size',
      help='Number of nodes in the first layer of the DNN',
      default=100,
      type=int
  )
  parser.add_argument(
      '--num-layers',
      help='Number of layers in the DNN',
      default=None,
      type=int
  )
  parser.add_argument(
        '--hidden-units',
        help="""\
             Hidden layer sizes to use for DNN feature columns, provided in comma-separated layers.
             If --scale-factor > 0, then only the size of the first layer will be used to compute
             the sizes of subsequent layers \
             """,
        default=[30, 15, 5]
  )
  parser.add_argument(
      '--layer-sizes-scale-factor',
      help='How quickly should the size of the layers in the DNN decay',
      default=0.7,
      type=float
  )
  parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
  )

  # Argument to turn on all logging
  parser.add_argument(
      '--verbosity',
      choices=[
          'DEBUG',
          'ERROR',
          'FATAL',
          'INFO',
          'WARN'
      ],
      default='INFO',
  )
  # Experiment arguments
  parser.add_argument(
      '--evaluate-after-sec',
      help="""Seconds between evaluation""",
      default=10,
      type=int
  )
  parser.add_argument(
      '--eval-steps',
      help='Number of steps to run evalution for at each checkpoint',
      default=100,
      type=int
  )


  args = parser.parse_args()
  
  if args.num_layers is not None:
    args.hidden_units = [
        max(2, int(args.first_layer_size *
                              args.layer_sizes_scale_factor**i))
                    for i in range(args.num_layers)
   ]
  
  
  # Set python level verbosity
  tf.logging.set_verbosity(args.verbosity)
  # Set C++ Graph Execution level verbosity
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
      tf.logging.__dict__[args.verbosity] / 10)

  # Run the training job
  hparams=hparam.HParams(**args.__dict__)
  run_experiment(hparams)
