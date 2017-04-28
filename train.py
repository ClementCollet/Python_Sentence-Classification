#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import pretrait
from modeleCNN import TextCNN
from tensorflow.contrib import learn

# Parametres
# Bade de Données
tf.flags.DEFINE_float("percent_validation", .1, "Pourcentage de donnée servant la validation")
tf.flags.DEFINE_string("database",1,"Choix de la base de donnée : 1 - Valence 2 - Subjectivité")
tf.flags.DEFINE_string("dossier_sortie","./runs/Valence/ParfaitW2v","Dossier de sortie contenant le vocabulaire, les summaries, les fichiers .txt ...")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/trainPOS.pos", "Adresse pour les données positive")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/trainNEG.neg", "Adresse pour les données negative")
tf.flags.DEFINE_string("objective_data_file", "./data/subjective/plot.tok.gt9.5000", "Adresse pour les données Objectives ")
tf.flags.DEFINE_string("subjective_data_file", "./data/subjective/quote.tok.gt9.5000", "Adresse pour les données Subjective")

# Parametres du prétraitement
tf.flags.DEFINE_boolean("word2vec",False, "Utilisationde Word2Vec ou embedding aléatoire")
tf.flags.DEFINE_boolean("stop_word",False, "Retrait des Stop Word")
tf.flags.DEFINE_boolean("token",False, "Tokenisation des mot")

# Parametres du modèle
tf.flags.DEFINE_integer("embedding_dim",50, "Dimension de la representation des mots")
tf.flags.DEFINE_string("filters", "3,4,5", "Taille des filtres")
tf.flags.DEFINE_integer("nb_filter",100, "Nombre de filtre par taille")

# Parametres pour l'entrainement
tf.flags.DEFINE_float("prob_dropout",0.5, "Probabilité de supprimmer un neurones")
tf.flags.DEFINE_float("l2_reg_lambda",3, "L2 regularisation lambda (default: 0.0)")
tf.flags.DEFINE_string("trainMethode","Adam", "Choix entre : Adam et Adadelta")
tf.flags.DEFINE_integer("batch_size",50, "Taille de batch")
tf.flags.DEFINE_integer("nb_iteration",13, "Nombre de fois où on passe sur les données d'entrainement")
tf.flags.DEFINE_integer("evaluate_every",50, "Evaluation toutes les X étapes")

# Parameters rapport à l'utilisation ou non du GPU
tf.flags.DEFINE_boolean("allow_soft_placement",True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement",False, "Log placement of ops on devices")


# Affiche les Parametres
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParametres:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
debut = datetime.datetime.now()

# Forcer une taille d'embedding de 300 si Word2Vec est activé
if FLAGS.word2vec:
    embedding_dimension = 300
else:
    embedding_dimension = FLAGS.embedding_dim

# Fonction d'entrainement et d'évalutation
def train_step(x_batch, y_batch):
    feed_dict = {cnn.input_x: x_batch,cnn.input_y: y_batch,cnn.prob_dropout: FLAGS.prob_dropout}
    _, step, summaries,prediction, loss, precision,input_argmax = sess.run([train_op, global_step, train_summary_op, cnn.predictions,cnn.loss, cnn.precision,cnn.input_argmax],feed_dict)
    time_str = datetime.datetime.now().isoformat(),

    # Calcul des différentes precision
    neg_predi_neg=0
    neg_predi_pos=0
    pos_predi_pos=0
    pos_predi_neg=0
    predi_pos_pos=0
    predi_pos_neg=0
    predi_neg_neg=0
    predi_neg_pos=0

    for i,val in enumerate(prediction):
        if input_argmax[i]==1:
            if val==1:
                pos_predi_pos+=1
            else:
                pos_predi_neg+=1
        else:
            if val==0:
                neg_predi_neg+=1
            else:
                neg_predi_pos+=1

        if val==1:
            if input_argmax[i]==1:
                predi_pos_pos+=1
            else:
                predi_pos_neg+=1
        else:
            if input_argmax[i]==0:
                predi_neg_neg+=1
            else:
                predi_neg_pos+=1

    if ((neg_predi_neg+neg_predi_pos)>0):
        pourcentage_neg_predi_neg=neg_predi_neg/(neg_predi_neg+neg_predi_pos)
    else:
        pourcentage_neg_predi_neg=-1
    if ((pos_predi_pos+pos_predi_neg)>0):
        pourcentage_pos_predi_pos=pos_predi_pos/(pos_predi_pos+pos_predi_neg)
    else:
        pourcentage_pos_predi_pos=-1

    if ((predi_neg_neg+predi_neg_pos)>0):
        pourcentage_predi_neg_neg=predi_neg_neg/(predi_neg_neg+predi_neg_pos)
    else:
        pourcentage_predi_neg_neg=-1
    if ((predi_pos_pos+predi_pos_neg)>0):
        pourcentage_predi_pos_pos=predi_pos_pos/(predi_pos_pos+predi_pos_neg)
    else:
        pourcentage_predi_pos_pos=-1

    if (step%50==0):
        print("{}: step {}, loss {:g}, acc {:g} NpN {:g} PpP {:g} pNN {:g} pPP {:g}".format(time_str, step, loss, precision,pourcentage_neg_predi_neg,pourcentage_pos_predi_pos,pourcentage_predi_neg_neg,pourcentage_predi_pos_pos))
    train_summary_writer.add_summary(summaries, step)
    return (step,precision,pourcentage_neg_predi_neg,pourcentage_pos_predi_pos,pourcentage_predi_pos_pos,pourcentage_predi_neg_neg)

def dev_step(x_batch, y_batch, writer=None):
    feed_dict = {cnn.input_x: x_batch,cnn.input_y: y_batch,cnn.prob_dropout: 1.0}
    step, summaries, prediction, loss, precision,input_argmax= sess.run([global_step, dev_summary_op,cnn.predictions,cnn.loss, cnn.precision,cnn.input_argmax], feed_dict)
    time_str = datetime.datetime.now().isoformat()

    # Calcul des différentes precision
    neg_predi_neg=0
    neg_predi_pos=0
    pos_predi_pos=0
    pos_predi_neg=0
    predi_pos_pos=0
    predi_pos_neg=0
    predi_neg_neg=0
    predi_neg_pos=0
    for i,val in enumerate(prediction):
        if input_argmax[i]==1:
            if val==1:
                pos_predi_pos+=1
            else:
                pos_predi_neg+=1
        else:
            if val==0:
                neg_predi_neg+=1
            else:
                neg_predi_pos+=1

        if val==1:
            if input_argmax[i]==1:
                predi_pos_pos+=1
            else:
                predi_pos_neg+=1
        else:
            if input_argmax[i]==0:
                predi_neg_neg+=1
            else:
                predi_neg_pos+=1

    if ((neg_predi_neg+neg_predi_pos)>0):
        pourcentage_neg_predi_neg=neg_predi_neg/(neg_predi_neg+neg_predi_pos)
    else:
        pourcentage_neg_predi_neg=-1
    if ((neg_predi_neg+neg_predi_pos)>0):
        pourcentage_pos_predi_pos=pos_predi_pos/(pos_predi_pos+pos_predi_neg)
    else:
        pourcentage_pos_predi_pos=-1

    if ((predi_neg_neg+predi_neg_pos)>0):
        pourcentage_predi_neg_neg=predi_neg_neg/(predi_neg_neg+predi_neg_pos)
    else:
        pourcentage_predi_neg_neg=-1
    if ((predi_pos_pos+predi_pos_neg)>0):
        pourcentage_predi_pos_pos=predi_pos_pos/(predi_pos_pos+predi_pos_neg)
    else:
        pourcentage_predi_pos_pos=-1

    print("{}:  acc {:g} NpN {:g} PpP {:g} pNN {:g} pPP {:g}".format(time_str, precision,pourcentage_neg_predi_neg,pourcentage_pos_predi_pos,pourcentage_predi_neg_neg,pourcentage_predi_pos_pos))

    if writer:
        writer.add_summary(summaries, step)

    return (step,precision,pourcentage_neg_predi_neg,pourcentage_pos_predi_pos,pourcentage_predi_pos_pos,pourcentage_predi_neg_neg)

# Chargement des données
if (FLAGS.database==1):
    x_text, y = pretrait.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file,FLAGS.stop_word,FLAGS.token)
else:
    x_text, y = pretrait.load_data_and_labels(FLAGS.objective_data_file, FLAGS.subjective_data_file,FLAGS.stop_word,FLAGS.token)

# Construction du dictionnaire
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))
print("Taille du Vocabulaire : {:d}".format(len(vocab_processor.vocabulary_)))

# Melange des données
np.random.seed(2)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Separation entrainement/evaluation
dev_sample_index = -1 * int(FLAGS.percent_validation * float(len(y)))
x_train, x_val = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_val = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

# Entrainement
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,log_device_placement=FLAGS.log_device_placement) # Gestion du CPU etc
    sess = tf.Session(config=session_conf)
    # Chargement du modele
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=embedding_dimension,
            filters=list(map(int, FLAGS.filters.split(","))),
            nb_filter=FLAGS.nb_filter,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        global_step = tf.Variable(0, name="global_step", trainable=False)

        # Choix de la rétropropagation
        if (FLAGS.trainMethode=="Adam"):
            optimizer = tf.train.AdamOptimizer(1e-3)
        else:
            optimizer = tf.train.AdadeltaOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Gestion des Summary
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Dosier de sortie
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir,FLAGS.dossier_sortie))
        print("Dossier de sortie {}\n".format(out_dir))

        # Summary pour l'erreur et les precision
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("precision", cnn.precision)

        # Summary pour l'entrainement
        train_summary_op = tf.summary.merge([loss_summary, acc_summary,grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Summary pour la validation
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Repertoire pour les checkpoints
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        # Ecriture du vocabulaire
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialisation
        sess.run(tf.global_variables_initializer())

        # 3 Fichiers, 1 avec toutes les precisions pour l'Entrainement, un autre pour l'évaluation, le dernier contient la precision apprentisasge VS Evaluation à tout moment
        adressfichierEval=FLAGS.dossier_sortie+"/RecapEval.txt"
        adressfichierTrain=FLAGS.dossier_sortie+"/RecapTrain.txt"
        adressfichierCompara=FLAGS.dossier_sortie+"/Comparatif.txt"
        fichierEval = open(adressfichierEval, "w")
        fichierTrain = open(adressfichierTrain, "w")
        fichierCompara=open(adressfichierCompara, "w")
        ecrit = ("Etape :       précision :        Négatif prédit Négatif :         Positif prédit Positif :        prédit Positif Positif :        prédit Négaif Négatif:      ")
        ecrit=str(ecrit)
        fichierEval.write(ecrit)
        fichierEval.write("\n")
        fichierTrain.write(ecrit)
        fichierTrain.write("\n")
        ecrit =  ("Etape :       précision Entrainement :     precision évaluation :     ")

        # Fichier pour les paramètres
        adressfichierParam=FLAGS.dossier_sortie+"/RecapParametre.txt"
        fichier_param = open(adressfichierParam,"w")
        for attr, value in sorted(FLAGS.__flags.items()):
            fichier_param.write(attr.upper()+"="+str(value))
            fichier_param.write("\n")

        # Chargement ou non d'un dictionnaire préentrainé
        if FLAGS.word2vec:
            vocabulary = vocab_processor.vocabulary_
            initW = None
            print("Chargement de word2vec{}".format("./data/GoogleNews-vectors-negative300.bin"))
            initW = pretrait.load_embedding_vectors_word2vec(vocabulary,"./data/GoogleNews-vectors-negative300.bin",True)
            print("Chargement Réussi")
            sess.run(cnn.W.assign(initW))

        # Boucle pour l'entrainement
        nb_batch = int(len(x_train)/FLAGS.batch_size)
        print("Il y aura",nb_batch-1," batch et ",FLAGS.nb_iteration," itérations sur la base de test.")
        for j in range(0,FLAGS.nb_iteration,1):
            # A chaque itérations, nous mélangeons nos données pour obtenir des batchs différents
            shuffle_indices = np.random.permutation(np.arange(len(y_train)))
            x_train = x_train[shuffle_indices]
            y_train = y_train[shuffle_indices]
            print("Itération numero ",j)
            for i in range(0,nb_batch-1,1):
                x_batch, y_batch = x_train[i*FLAGS.batch_size+i:(i+1)*FLAGS.batch_size+i+1], y_train[i*FLAGS.batch_size+i:(i+1)*FLAGS.batch_size+i+1]
                res=train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)

                # Nous n'evaluons pas à toutes les étapes
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    sort=dev_step(x_val, y_val, writer=dev_summary_writer)
                    print("")
                    ecrit = (sort[0],round(sort[1],2),round(sort[2],2),round(sort[3],2),round(sort[4],2),round(sort[5],2))
                    ecrit=str(ecrit)
                    fichierEval.write(ecrit)
                    fichierEval.write("\n")
                    ecrit = (res[0],round(res[1],2),round(res[2],2),round(res[3],2),round(res[4],2),round(res[5],2))
                    ecrit=str(ecrit)
                    fichierTrain.write(ecrit)
                    fichierTrain.write("\n")
                    ecrit=(sort[0],round(res[1],2),round(sort[1],2))
                    ecrit=str(ecrit)
                    fichierCompara.write(ecrit)
                    fichierCompara.write("\n")


        path = saver.save(sess, checkpoint_prefix)
        print("Poids enregistrés dans {}\n".format(path))

        print("Entrainement fini")

        print("Evaluation Finale :")

        # Ecriture et fermeture des fichiers
        sort=dev_step(x_val, y_val, writer=dev_summary_writer)
        ecrit = (sort[0],";",round(sort[1],2),";",round(sort[2],2),";",round(sort[3],2),";",round(sort[4],2),";",round(sort[5],2))
        ecrit=str(ecrit)
        fichierEval.write(ecrit)
        fichierEval.write("\n")
        fichierEval.write("C'était l'Evaluation Finale")
        fichierEval.write("\n")

        fin = datetime.datetime.now()
        duree = fin-debut
        duree =(duree.seconds/(60))
        minute= str(duree)
        fichier_param.write(minute)
        fichier_param.write(" minutes")
        fichierEval.close()
        fichierTrain.close()
        fichier_param.close()
        fichierCompara.close()
