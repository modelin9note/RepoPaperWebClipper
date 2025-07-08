Sequence to Sequence Learning
with Neural Networks

Deep Neural Networks (DNNs) are powerful models that have achieved excellent
performance on difficult learning tasks. Although DNNs work well whenever
large labeled training sets are available, they cannot be used to map sequences to
sequences. In this paper, we present a general end-to-end approach to sequence
learning that makes minimal assumptions on the sequence structure. Our method
uses a multilayered Long Short-TermMemory (LSTM) to map the input sequence
to a vector of a fixed dimensionality, and then another deep LSTM to decode the
target sequence from the vector. Our main result is that on an English to French
translation task fromtheWMT’14 dataset, the translations produced by the LSTM
achieve a BLEU score of 34.8 on the entire test set, where the LSTM’s BLEU
score was penalized on out-of-vocabulary words. Additionally, the LSTM did not
have difficulty on long sentences. For comparison, a phrase-based SMT system
achieves a BLEU score of 33.3 on the same dataset. When we used the LSTM
to rerank the 1000 hypotheses produced by the aforementioned SMT system, its
BLEU score increases to 36.5, which is close to the previous best result on this
task. The LSTM also learned sensible phrase and sentence representations that
are sensitive to word order and are relatively invariant to the active and the passive
voice. Finally, we found that reversing the order of the words in all source
sentences (but not target sentences) improved the LSTM’s performancemarkedly,
because doing so introduced many short term dependencies between the source
and the target sentence which made the optimization problem easier.

1 Introduction
Deep Neural Networks (DNNs) are extremely powerful machine learning models that achieve excellent
performance on difficult problems such as speech recognition [13, 7] and visual object recognition
[19, 6, 21, 20]. DNNs are powerful because they can perform arbitrary parallel computation
for a modest number of steps. A surprising example of the power of DNNs is their ability to sort
N N-bit numbers using only 2 hidden layers of quadratic size [27]. So, while neural networks are
related to conventional statistical models, they learn an intricate computation. Furthermore, large
DNNs can be trained with supervised backpropagationwhenever the labeled training set has enough
information to specify the network’s parameters. Thus, if there exists a parameter setting of a large
DNN that achieves good results (for example, because humans can solve the task very rapidly),
supervised backpropagation will find these parameters and solve the problem.

1 Introduction
Deep Neural Networks (DNNs) are extremely powerful machine learning models that achieve excellent
performance on difficult problems such as speech recognition [13, 7] and visual object recognition
[19, 6, 21, 20]. DNNs are powerful because they can perform arbitrary parallel computation
for a modest number of steps. A surprising example of the power of DNNs is their ability to sort
N N-bit numbers using only 2 hidden layers of quadratic size [27]. So, while neural networks are
related to conventional statistical models, they learn an intricate computation. Furthermore, large
DNNs can be trained with supervised backpropagationwhenever the labeled training set has enough
information to specify the network’s parameters. Thus, if there exists a parameter setting of a large
DNN that achieves good results (for example, because humans can solve the task very rapidly),
supervised backpropagation will find these parameters and solve the problem.

Despite their flexibility and power, DNNs can only be applied to problems whose inputs and targets
can be sensibly encoded with vectors of fixed dimensionality. It is a significant limitation, since
many important problems are best expressed with sequences whose lengths are not known a-priori.
For example, speech recognition and machine translation are sequential problems. Likewise, question
answering can also be seen as mapping a sequence of words representing the question to a
1
sequence of words representing the answer. It is therefore clear that a domain-independent method
that learns to map sequences to sequences would be useful.

Sequences pose a challenge for DNNs because they require that the dimensionality of the inputs and
outputs is known and fixed. In this paper, we show that a straightforward application of the Long
Short-Term Memory (LSTM) architecture [16] can solve general sequence to sequence problems.
The idea is to use one LSTMto read the input sequence, one timestep at a time, to obtain large fixeddimensional
vector representation, and then to use another LSTM to extract the output sequence
from that vector (fig. 1). The second LSTMis essentially a recurrent neural network languagemodel
[28, 23, 30] except that it is conditioned on the input sequence. The LSTM’s ability to successfully
learn on data with long range temporal dependencies makes it a natural choice for this application
due to the considerable time lag between the inputs and their corresponding outputs (fig. 1).

There have been a number of related attempts to address the general sequence to sequence learning
problem with neural networks. Our approach is closely related to Kalchbrenner and Blunsom [18]
who were the first to map the entire input sentence to vector, and is related to Cho et al. [5] although
the latter was used only for rescoring hypotheses produced by a phrase-based system. Graves [10]
introduced a novel differentiable attention mechanism that allows neural networks to focus on different
parts of their input, and an elegant variant of this idea was successfully applied to machine
translation by Bahdanau et al. [2]. The Connectionist Sequence Classification is another popular
technique for mapping sequences to sequences with neural networks, but it assumes a monotonic
alignment between the inputs and the outputs [11].

There have been a number of related attempts to address the general sequence to sequence learning
problem with neural networks. Our approach is closely related to Kalchbrenner and Blunsom [18]
who were the first to map the entire input sentence to vector, and is related to Cho et al. [5] although
the latter was used only for rescoring hypotheses produced by a phrase-based system. Graves [10]
introduced a novel differentiable attention mechanism that allows neural networks to focus on different
parts of their input, and an elegant variant of this idea was successfully applied to machine
translation by Bahdanau et al. [2]. The Connectionist Sequence Classification is another popular
technique for mapping sequences to sequences with neural networks, but it assumes a monotonic
alignment between the inputs and the outputs [11].