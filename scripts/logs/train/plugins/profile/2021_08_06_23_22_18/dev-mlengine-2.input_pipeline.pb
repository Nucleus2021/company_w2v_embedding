	"rl=C??"rl=C??!"rl=C??	??D?"&@??D?"&@!??D?"&@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$"rl=C???%Tpx??A\r?)???Y??g???*	??n??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorRԙ{H?$@![j?O?X@)Rԙ{H?$@1[j?O?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?????!5???\??)?????15???\??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelisml?V^????!&"?|??)?E'K????15?-???:Preprocessing2F
Iterator::Model?dV?p;??!	?????)$?????w?1 om?eU??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap|???$@!0?=>??X@)??? ?Y?1:???ǽ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 11.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2t10.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??D?"&@Ibi?~?;V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?%Tpx???%Tpx??!?%Tpx??      ??!       "      ??!       *      ??!       2	\r?)???\r?)???!\r?)???:      ??!       B      ??!       J	??g?????g???!??g???R      ??!       Z	??g?????g???!??g???b      ??!       JCPU_ONLYY??D?"&@b qbi?~?;V@