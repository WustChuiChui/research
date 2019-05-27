
import os,sys,types
import tensorflow as tf
import traceback
import argparse


def save_model(checkpoint_prefix, output_node):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        restore_saver = tf.train.import_meta_graph(checkpoint_prefix + '.meta')
        restore_saver.restore(sess, checkpoint_prefix)
        out_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node.split(","))
        tf.train.write_graph(out_graph_def, "",  "graph.pb", as_text=False)
        
        op = sess.graph.get_tensor_by_name("viterbi_seq_out:0")
        input_x = sess.graph.get_tensor_by_name('input_x:0')
        keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
        sequence_lengths = sess.graph.get_tensor_by_name('sequence_lengths:0')
        feed_dic ={input_x: [[59, 2713, 62, 2303, 20, 30, 69, 10, 11, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
                   keep_prob:1.0,
                   sequence_lengths: [30]}
        print(sess.run(op, feed_dic))                                    


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_prefix", default="model/model/ckpt", type=str, help="checkpoint_prefix")
parser.add_argument("--output_node", default="viterbi_seq_out", type=str, help="")
args = parser.parse_args()
save_model(args.checkpoint_prefix, args.output_node)

