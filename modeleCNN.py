import tensorflow as tf
import numpy as np


class TextCNN(object):
    def __init__(self, sequence_length, num_classes, vocab_size,embedding_size, filters, nb_filter, l2_reg_lambda=0.0):

        # Placeholders (propre à tensorflow) pour l'entrée et la sortie
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int64, [None, num_classes], name="input_y")
        self.prob_dropout = tf.placeholder(tf.float32, name="prob_dropout")
        l2_loss = tf.constant(0.0)

        # Couche Embedding
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Convolution + maxpool pour chaque taille de filtre
        pooled_outputs = []
        for i, filter_size in enumerate(filters):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Couche de Convolution
                filter_shape = [filter_size, embedding_size, 1, nb_filter]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[nb_filter]), name="b")
                conv = tf.nn.conv2d(self.embedded_chars_expanded,W,strides=[1, 1, 1, 1],padding="VALID",name="conv") # Padding correspond à la gestion des bords
                # Ajout de b et relu
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling sur les sorties
                pooled = tf.nn.max_pool(h,ksize=[1, sequence_length - filter_size + 1, 1, 1],strides=[1, 1, 1, 1],padding='VALID',name="pool")
                pooled_outputs.append(pooled)

        # on rassemble les motifs detectés
        nb_filter_total = nb_filter * len(filters)
        self.h_pool = tf.concat(axis=3, values=pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, nb_filter_total])

        # Dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.prob_dropout)

        # Scores et predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[nb_filter_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calcul de l'erreur avec la cross-entropy
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        self.input_argmax=tf.argmax(self.input_y, 1)

        # Precision
        with tf.name_scope("precision"):
            correct_predictions = tf.equal(self.predictions,self.input_argmax)
            self.precision = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="precision")
