import sys
sys.path.append("../")
from common.baseLayer import layerNormalization
import tensorflow as tf


def attention(rnn_outputs, keep_dropout):
    fw_outputs, bw_outputs = rnn_outputs
    hidden_size = int(fw_outputs.shape[2]) #tensor -> int
    W = tf.Variable(tf.random_normal([hidden_size], stddev=0.1))
    H = fw_outputs + bw_outputs  # (batch_size, seq_len, hidden_size)
    M = tf.tanh(H)  # (batch_size, seq_len, hidden_size)
    out_size = H.get_shape()[1]

    h = tf.matmul(tf.reshape(M, [-1, hidden_size]),tf.reshape(W, [-1, 1]))
    alpha = tf.nn.softmax(tf.reshape(h,(-1, out_size)))  # (batch_size, seq_len)
    r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(alpha, [-1, out_size, 1]))
    r = tf.squeeze(r)
    h_star = tf.tanh(r)  # (batch_size , hidden_size)
    h_drop = tf.nn.dropout(h_star, keep_dropout)

    return h_drop

def Attention(inputs, attention_size, time_major=False):
    if isinstance(inputs, tuple):
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
    return output

def multiheadAttention(queries, keys, num_units=None, num_heads=8, dropout_rate=0, \
                        is_training=True, causality=False, scope="multihead_attention", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:  # set default size for attention size C
            num_units = queries.get_shape().as_list()[-1]

        # Linear Projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # [N, T_q, C]
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # [N, T_k, C]
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # [N, T_k, C]

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0)  # [num_heads * N, T_q, C/num_heads]
        K_ = tf.concat(tf.split(K, num_heads, axis=-1), axis=0)  # [num_heads * N, T_k, C/num_heads]
        V_ = tf.concat(tf.split(V, num_heads, axis=-1), axis=0)  # [num_heads * N, T_k, C/num_heads]

        # Attention
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (num_heads * N, T_q, T_k)

        # Scale : outputs = outputs / sqrt( d_k)
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        # see : https://github.com/Kyubyong/transformer/issues/3
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)  # -infinity
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation: outputs is a weight matrix
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # reshape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # residual connection
        outputs += queries

        # layer normaliztion
        outputs = layerNormalization(outputs)
        return outputs

def feedforward(inputs, num_units=[2048, 512], scope="multihead_attention", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        print("Conv ret:", outputs.shape)
        # Residual connection
        outputs += inputs

        # Normalize
        outputs = layerNormalization(outputs)
    return outputs
