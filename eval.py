#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import pretrait
from modeleCNN import TextCNN
from tensorflow.contrib import learn
import csv

def dev_step(x_batch, y_batch, writer=None):
    feed_dict = {cnn.input_x: x_batch,cnn.input_y: y_batch,cnn.prob_dropout: 1.0}
    step, summaries, prediction, loss, precision,input_argmax= sess.run([global_step, dev_summary_op,cnn.predictions,cnn.loss, cnn.precision,cnn.input_argmax], feed_dict)
    time_str = datetime.datetime.now().isoformat()
    negReussi=0
    negLoupe=0
    posReussi=0
    posLoupe=0
    for i,val in enumerate(prediction):
        if input_argmax[i]==1:
            if val==1:
                posReussi+=1
            else:
                posLoupe+=1
        else:
            if val==0:
                negReussi+=1
            else:
                negLoupe+=1
    pourcentageNegReussi=negReussi/(negReussi+negLoupe)
    pourcentagePosReussi=posReussi/(posReussi+posLoupe)
    if (step%10==0):
        print("{}: step {}, loss {:g}, acc {:g} Pourcentage Negatif Reussi {:g} Pourcentage Positif Reussi {:g}  ".format(time_str, step, loss, precision,pourcentageNegReussi,pourcentagePosReussi))
    if writer:
        writer.add_summary(summaries, step)
    return (pourcentageNegReussi,pourcentagePosReussi)

# Parameters
# ==================================================

# Base de données
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/evalPOS.pos", "Adresse pour les données positive".")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/evalNEG.neg", "Adresse pour les données negative"")

# Parametres d'évaluation
tf.flags.DEFINE_string("checkpoint_dir", "", "Dossier avec les checkpoints à fournir lors de l'appel")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

#Chargement des données
x_raw, y_test = pretrait.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
y_test = np.argmax(y_test, axis=1)

# Chargement du dictionnaire
print(FLAGS.checkpoint_dir)
vocab_path = os.path.join(FLAGS.checkpoint_dir, "vocab")
print(vocab_path)
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluation\n")

# Evaluation
# ==================================================
checkpoint_file = FLAGS.checkpoint_dir+"\checkpoints\model"
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Chargemnt du modèle et de ses poids
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Copie des placeholder
        input_x = graph.get_operation_by_name("input_x").outputs[0]

        dropout_keep_prob = graph.get_operation_by_name("prob_dropout").outputs[0]

        # Tensor que nous voulons récuperer de notre modèle
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Récupération du vecteur des prédictions
        all_predictions =sess.run(predictions, {input_x: x_test, dropout_keep_prob: 1.0})

# Affichage de la precision
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Nombre d'avis testé: {}".format(len(y_test)))
    print("Precision : {:g}".format(correct_predictions/float(len(y_test))))

# Enrgistrement des résultats dans un fichier
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "prediction.csv")
print("Dossier d'enregistrement {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
