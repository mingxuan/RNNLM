
from __future__ import print_function
import logging
import pprint
import math
import numpy
import traceback
import operator

import theano
from six.moves import input
from picklable_itertools.extras import equizip
from theano import tensor

from blocks.bricks import Tanh, Initializable
from blocks.bricks.base import application
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import SimpleRecurrent, Bidirectional
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.parallel import Fork
from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout, SoftmaxEmitter, LookupFeedback)
from blocks.config import config
from blocks.graph import ComputationGraph
from fuel.transformers import Mapping, Batch, Padding, Filter
from fuel.datasets import OneBillionWord, TextFile
from fuel.schemes import ConstantScheme
from blocks.serialization import load_parameter_values
from blocks.algorithms import (GradientDescent, Scale,
                               StepClipping, CompositeRule)
from blocks.initialization import Orthogonal, IsotropicGaussian, Constant
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Printing, Timing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.filter import VariableFilter
from blocks.utils import dict_union

from blocks.search import BeamSearch


class Rnnlm(Initializable):
    def __init__(self, vocab_size, dimension, n_hids, **kwargs):
        super(Rnnlm, self).__init__(**kwargs)
        transition = SimpleRecurrent(activation=Tanh(),
                                     dim=n_hids, name='transition')
        read_out = Readout(readout_dim=vocab_size,
                           source_names=[transition.apply.states[0]],
                           emitter=SoftmaxEmitter(name="emitter"),
                           feedback_brick=LookupFeedback(vocab_size, dimension),
                           name='readout')

        generator = SequenceGenerator(readout=read_out,
                                      transition=transition,
                                      name='generator')

        self.generator = generator
        self.children = [self.generator]

    @application(inputs=['sentence','sentence_mask'],
                 outputs=['cost'])
    def cost(self, sentence, sentence_mask):
        sentence = sentence.T
        sentence_mask = sentence_mask.T

        cost_matrix = self.generator.cost_matrix(mask=sentence_mask,
                                                    outputs=sentence)

        return cost_matrix

def main():

    import configurations
    from stream import DStream
    logger = logging.getLogger(__name__)
    cfig = getattr(configurations, 'get_config_penn')()

    rnnlm = Rnnlm(cfig['vocabsize'], cfig['nemb'], cfig['nhids'])
    rnnlm.weights_init = IsotropicGaussian(0.1)
    rnnlm.biases_init = Constant(0.)
    rnnlm.push_initialization_config()
    rnnlm.generator.transition.weights_init = Orthogonal()

    sentence = tensor.lmatrix('sentence')
    sentence_mask = tensor.matrix('sentence_mask')
    batch_cost = rnnlm.cost(sentence, sentence_mask).sum()
    batch_size = sentence.shape[1].copy(name='batch_size')
    cost = aggregation.mean(batch_cost, batch_size)
    cost.name = "sequence_log_likelihood"
    logger.info("Cost graph is built")

    model = Model(cost)
    parameters = model.get_parameter_dict()
    logger.info("Parameters:\n" +
                pprint.pformat(
                    [(key, value.get_value().shape) for key, value
                        in parameters.items()],
                    width=120))

    for brick in model.get_top_bricks():
        brick.initialize()
    cg = ComputationGraph(cost)
    algorithm = GradientDescent(
        cost=cost, parameters=cg.parameters,
        step_rule=CompositeRule([StepClipping(10.0), Scale(0.01)]))

    gradient_norm = aggregation.mean(algorithm.total_gradient_norm)
    step_norm = aggregation.mean(algorithm.total_step_norm)
    monitored_vars = [cost, gradient_norm, step_norm]

    train_monitor = TrainingDataMonitoring(variables=monitored_vars, after_batch=True,
                                           before_first_epoch=True, prefix='tra')

    extensions = [train_monitor, Timing(), Printing(after_batch=True),
                  FinishAfter(after_n_epochs=1000),
                  Printing(every_n_batches=1)]

    train_stream = DStream(datatype='train', config=cfig)
    main_loop = MainLoop(model=model,
                         data_stream=train_stream,
                         algorithm=algorithm,
                         extensions=extensions)

    main_loop.run()

if __name__=='__main__':
    main()
