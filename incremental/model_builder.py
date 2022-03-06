# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import base_ops
import numpy as np


def compute_vertex_channels(input_channels, output_channels, matrix):
    """Computes the number of channels at every vertex.
  Given the input channels and output channels, this calculates the number of
  channels at each interior vertex. Interior vertices have the same number of
  channels as the max of the channels of the vertices it feeds into. The output
  channels are divided amongst the vertices that are directly connected to it.
  When the division is not even, some vertices may receive an extra channel to
  compensate.
  Args:
    input_channels: input channel count.
    output_channels: output channel count.
    matrix: adjacency matrix for the module (pruned by model_spec).
  Returns:
    list of channel counts, in order of the vertices.
  """
    num_vertices = np.shape(matrix)[0]

    vertex_channels = [0] * num_vertices
    vertex_channels[0] = input_channels
    vertex_channels[num_vertices - 1] = output_channels

    if num_vertices == 2:
        # Edge case where module only has input and output vertices
        return vertex_channels

    # Compute the in-degree ignoring input, axis 0 is the src vertex and axis 1 is
    # the dst vertex. Summing over 0 gives the in-degree count of each vertex.
    in_degree = np.sum(matrix[1:], axis=0)
    interior_channels = output_channels // in_degree[num_vertices - 1]
    correction = output_channels % in_degree[num_vertices - 1]  # Remainder to add

    # Set channels of vertices that flow directly to output
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            vertex_channels[v] = interior_channels
            if correction:
                vertex_channels[v] += 1
                correction -= 1

    # Set channels for all other vertices to the max of the out edges, going
    # backwards. (num_vertices - 2) index skipped because it only connects to
    # output.
    for v in range(num_vertices - 3, 0, -1):
        if not matrix[v, num_vertices - 1]:
            for dst in range(v + 1, num_vertices - 1):
                if matrix[v, dst]:
                    vertex_channels[v] = max(vertex_channels[v], vertex_channels[dst])
        assert vertex_channels[v] > 0

    # tf.logging.info('vertex_channels: %s', str(vertex_channels))

    # Sanity check, verify that channels never increase and final channels add up.
    final_fan_in = 0
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            final_fan_in += vertex_channels[v]
        for dst in range(v + 1, num_vertices - 1):
            if matrix[v, dst]:
                assert vertex_channels[v] >= vertex_channels[dst]
    assert final_fan_in == output_channels or num_vertices == 2
    # num_vertices == 2 means only input/output nodes, so 0 fan-in

    return vertex_channels


def projection(channels, is_training, data_format):
    """1x1 projection (as in ResNet) followed by batch normalization and ReLU."""
    with tf.compat.v1.variable_scope('projection'):
        net = base_ops.Conv_BN_ReLU(1, channels, is_training, data_format)
    return net


def truncate(inputs_shape, inputs, channels, data_format):
    """Slice the inputs to channels if necessary."""
    if data_format == 'channels_last':
        input_channels = inputs_shape[3]
    else:
        assert data_format == 'channels_first'
        input_channels = inputs_shape[1]

    if input_channels < channels:
        raise ValueError('input channel < output channels for truncate')
    elif input_channels == channels:
        return inputs  # No truncation necessary
    else:
        # Truncation should only be necessary when channel division leads to
        # vertices with +1 channels. The input vertex should always be projected to
        # the minimum channel count.
        assert input_channels - channels == 1
        if data_format == 'channels_last':
            return tf.slice(inputs, [0, 0, 0, 0], [-1, -1, -1, channels])
        else:
            return tf.slice(inputs, [0, 0, 0, 0], [-1, channels, -1, -1])


'''
def build_module(spec, inputs, channels, is_training):
    """Build a custom module using a proposed model spec.
  Builds the model using the adjacency matrix and op labels specified. Channels
  controls the module output channel count but the interior channels are
  determined via equally splitting the channel count whenever there is a
  concatenation of Tensors.
  Args:
    spec: ModelSpec object.
    inputs: input Tensors to this module.
    channels: output channel count.
    is_training: bool for whether this model is training.
  Returns:
    output Tensor from built module.
  Raises:
    ValueError: invalid spec
  """
    num_vertices = np.shape(spec.matrix)[0]

    if spec.data_format == 'channels_last':
        channel_axis = 3
    elif spec.data_format == 'channels_first':
        channel_axis = 1
    else:
        raise ValueError('invalid data_format')

    input_channels = inputs.get_shape()[channel_axis]
    # vertex_channels[i] = number of output channels of vertex i
    vertex_channels = compute_vertex_channels(
        input_channels, channels, spec.matrix)

    # Construct tensors from input forward
    #tensors = [tf.identity(inputs, name='input')]

    x = tf.keras.layers.Lambda(lambda a: a, name='input')(inputs)
    tensors = [x]

    final_concat_in = []
    for t in range(1, num_vertices - 1):
        with tf.compat.v1.variable_scope('vertex_{}'.format(t)):
            # Create interior connections, truncating if necessary
            add_in = [truncate(tensors[src], vertex_channels[t], spec.data_format)
                      for src in range(1, t) if spec.matrix[src, t]]

            # Create add connection from projected input
            if spec.matrix[0, t]:
                add_in.append(projection(
                    tensors[0],
                    vertex_channels[t],
                    is_training,
                    spec.data_format))

            if len(add_in) == 1:
                vertex_input = add_in[0]
            else:
                vertex_input = tf.add_n(add_in)

            # Perform op at vertex t
            op = base_ops.OP_MAP[spec.ops[t]](
                is_training=is_training,
                data_format=spec.data_format)
            vertex_value = op.build(vertex_input, vertex_channels[t])

        tensors.append(vertex_value)
        if spec.matrix[t, num_vertices - 1]:
            final_concat_in.append(tensors[t])

    # Construct final output tensor by concating all fan-in and adding input.
    if not final_concat_in:
        # No interior vertices, input directly connected to output
        assert spec.matrix[0, num_vertices - 1]
        with tf.compat.v1.variable_scope('output'):
            outputs = projection(
                tensors[0],
                channels,
                is_training,
                spec.data_format)

    else:
        if len(final_concat_in) == 1:
            outputs = final_concat_in[0]
        else:
            #outputs = tf.concat(final_concat_in, channel_axis)
            outputs = tf.keras.layers.Concatenate(channel_axis)(final_concat_in)
        if spec.matrix[0, num_vertices - 1]:
            outputs += projection(
                tensors[0],
                channels,
                is_training,
                spec.data_format)

    outputs = tf.identity(outputs, name='output')
    return outputs
'''


class Cell_Model(tf.keras.Model):
    """
    If the stride is not equal to 1 or the filters of the input is not equal to given filter_nums, then it will need a
    Con1x1 layer with given stride to project the input.
    """

    def __init__(self, spec, inputs_shape, channels, is_training):
        super(Cell_Model, self).__init__()

        self.inputs_shape = inputs_shape
        self.spec = spec
        self.is_training = is_training
        self.channels = channels
        self.num_vertices = np.shape(spec.matrix)[0]
        if spec.data_format == 'channels_last':
            self.channel_axis = 3
        elif spec.data_format == 'channels_first':
            self.channel_axis = 1
        else:
            raise ValueError('invalid data_format')

        input_channels = inputs_shape[self.channel_axis]
        # vertex_channels[i] = number of output channels of vertex i
        self.vertex_channels = compute_vertex_channels(
            input_channels, channels, spec.matrix)

        # --------------------------------------------------------
        self.ops = {}
        self.proj_list = {}

        final_concat_in = []
        # Construct tensors shape from input forward
        self.tensors = [inputs_shape]

        for t in range(1, self.num_vertices - 1):
            with tf.compat.v1.variable_scope('vertex_{}'.format(t)):

                # Create add connection from projected input
                if self.spec.matrix[0, t]:
                    self.proj_list[t] = projection(
                        self.vertex_channels[t],
                        self.is_training,
                        self.spec.data_format)

                    # Perform op at vertex t
                op = base_ops.OP_MAP[self.spec.ops[t]](
                    is_training=self.is_training,
                    data_format=self.spec.data_format)
                self.ops[t] = op.build(self.vertex_channels[t])

            t_shape = list(inputs_shape)
            t_shape[self.channel_axis] = self.vertex_channels[t]

            self.tensors.append(tuple(t_shape))
            if self.spec.matrix[t, self.num_vertices - 1]:
                final_concat_in.append(self.tensors[t])

        # Construct final output tensor by concating all fan-in and adding input.
        if not final_concat_in:
            # No interior vertices, input directly connected to output
            assert spec.matrix[0, self.num_vertices - 1]
            with tf.compat.v1.variable_scope('output'):
                self.outputs1 = projection(
                    self.channels,
                    self.is_training,
                    self.spec.data_format)

        else:
            if self.spec.matrix[0, self.num_vertices - 1]:
                self.outputs1 = projection(
                    self.channels,
                    self.is_training,
                    self.spec.data_format)

    def call(self, inputs):
        # Construct tensors from input forward
        tensors = [tf.identity(inputs, name='input')]
        final_concat_in = []

        for t in range(1, self.num_vertices - 1):
            with tf.compat.v1.variable_scope('vertex_{}'.format(t)):
                # Create interior connections, truncating if necessary
                add_in = [truncate(self.tensors[src], tensors[src], self.vertex_channels[t], self.spec.data_format)
                          for src in range(1, t) if self.spec.matrix[src, t]]

                # Create add connection from projected input
                if self.spec.matrix[0, t]:
                    add_in.append(self.proj_list[t](tensors[0]))

                if len(add_in) == 1:
                    vertex_input = add_in[0]
                else:
                    vertex_input = tf.add_n(add_in)

                # Perform op at vertex t
                vertex_value = self.ops[t](vertex_input)

            tensors.append(vertex_value)
            if self.spec.matrix[t, self.num_vertices - 1]:
                final_concat_in.append(tensors[t])

        # Construct final output tensor by concating all fan-in and adding input.
        if not final_concat_in:
            # No interior vertices, input directly connected to output
            assert spec.matrix[0, self.num_vertices - 1]
            with tf.compat.v1.variable_scope('output'):
                outputs = self.outputs1(tensors[0])
        else:
            if len(final_concat_in) == 1:
                outputs = final_concat_in[0]
            else:
                outputs = tf.concat(final_concat_in, self.channel_axis)

            if self.spec.matrix[0, self.num_vertices - 1]:
                outputs += self.outputs1(tensors[0])

        return outputs

    def build_graph(self):
        shape = tuple(list(self.inputs_shape)[1:])
        x = tf.keras.Input(shape=shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


if __name__ == '__main__':
    from model_spec import ModelSpec
    import keras

    matrix = np.array([[0, 1, 1, 1, 0, 1, 0],  # input layer
                       [0, 0, 0, 0, 0, 0, 1],  # 1x1 conv
                       [0, 0, 0, 0, 0, 0, 1],  # 3x3 conv
                       [0, 0, 0, 0, 1, 0, 0],  # 5x5 conv (replaced by two 3x3's)
                       [0, 0, 0, 0, 0, 0, 1],  # 5x5 conv (replaced by two 3x3's)
                       [0, 0, 0, 0, 0, 0, 1],  # 3x3 max-pool
                       [0, 0, 0, 0, 0, 0, 0]])

    ops = ['INPUT', 'conv3x3-bn-relu', 'maxpool3x3', 'conv1x1-bn-relu', 'maxpool3x3', 'identity',
           'OUTPUT']

    spec = ModelSpec(matrix, ops)

    model = Cell_Model(spec, (None, 28, 28, 1), channels=64, is_training=True)
    print(model.build_graph().summary())
    tf.keras.utils.plot_model(
        model.build_graph(), to_file='model.png', show_shapes=True, show_dtype=False,
        show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96,
        layer_range=None, show_layer_activations=False
    )
