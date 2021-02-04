Tareas de llenado de espacios y detección de intención de comprensión del lenguaje hablado
Modelos básicos para llenado de ranuras y detección de intención:
Una implementación para la parte de "enfoque" del artículo "Codificador-decodificador con mecanismo de enfoque para la comprensión del lenguaje hablado basado en el etiquetado de secuencias" (Su Zhu y Kai Yu, 2016) .
Una implementación de BLSTM-CRF basada en jiesutd / NCRFpp
Una implementación de entrenamiento conjunto de tareas de llenado de espacios y detección de intenciones (Bing Liu e Ian Lane, 2016) .
Modelos básicos + ELMo / BERT / XLNET
Tutoriales sobre conjuntos de datos:
ATIS
SNIPS
El conjunto de datos multilingüe de Facebook (inglés / español / tailandés)
MIT_Restaurant_Movie_corpus (sin intención)
Asistente de compras de comercio electrónico (ECSA) de Alibaba (sin intención, en chino)
CoNLL-2003 NER (sin intención)
anotación de datos

Sección	Descripción
Preparar	Paquetes requeridos
Evaluación de la detección de intenciones para múltiples intenciones	Cómo informar el rendimiento de la detección de intenciones en el conjunto de datos ATIS
Tutoriales A: con incrustaciones de palabras previamente entrenadas	Tutoriales A: llenado de espacios y detección de intención con incrustaciones de palabras previamente entrenadas
Tutoriales B: con ELMo	Tutoriales B: Llenado de ranuras y detección de intención con ELMo
Tutoriales C: con BERT	Tutoriales C: Llenado de ranuras y detección de intención con BERT
Tutoriales D: con XLNET	Tutoriales D: Llenado de ranuras y detección de intención con XLNET
Resultados	Resultados de diferentes métodos en ciertos conjuntos de datos
Modo de inferencia	Modo de inferencia
Referencia	¿Cómo citar?
Preparar
Python 3.6.x
pytorch 1.4.0
pip install gpustat [si se usa gpu]
embeddings : pip install embeddings
ELMo en allennlp : pip install allennlp
BERT / XLNET en transformadores : pip install transformers (2.5.0)
Acerca de las evaluaciones de detección de intenciones en conjuntos de datos ATIS y SNIPS
Como podemos saber por los conjuntos de datos, ATIS puede tener múltiples intenciones para un enunciado, mientras que SNIPS solo tiene un intento para un enunciado. Por ejemplo, "muéstrame todos los vuelos y tarifas de denver a san francisco <=> atis_flight && atis_airfare". Por lo tanto, existe un truco público en las etapas de capacitación y evaluación para la detección de intenciones del conjunto de datos ATIS.

# f03c15¡¡¡NOTA!!! # f03c15: Impactado por el artículo "¿Qué queda por entender en ATIS?" , casi todos los trabajos sobre ATIS eligen la primera intención como etiqueta para entrenar un clasificador de intenciones "softmax". En la etapa de evaluación, se considerará correcto si la intención prevista es una de las múltiples intenciones.

QUE HACER:
Agregar char-embeddings
Tutoriales A: llenado de espacios y detección de intención con incrustaciones de palabras previamente entrenadas
Las incrustaciones de palabras previamente entrenadas se toman prestadas de los modelos de lenguaje CNN-BLSTM de ELMo, donde las incrustaciones de palabras son modeladas por char-CNN. Extraemos las inclusiones de palabras pretrained para ATIS , SNIPS , conjunto de datos multilingüe del Facebook y MIT_Restaurant_Movie_corpus conjuntos de datos (E / S) w intención de:
  scripts de python3 / get_ELMo_word_embedding_for_a_dataset.py \
          --in_files data / atis-2 / {entrenar, válido, prueba} \
          --output_word2vec local / word_embeddings / elmo_1024_cased_for_atis.txt
  scripts de python3 / get_ELMo_word_embedding_for_a_dataset.py \
          --in_files data / snips / {entrenar, válido, prueba} \
          --output_word2vec local / word_embeddings / elmo_1024_cased_for_snips.txt
  scripts de python3 / get_ELMo_word_embedding_for_a_dataset.py \
          --in_files data / MIT_corpus / {movie_eng, movie_trivia10k13, restaurant} / {train, valid, test} \
          --output_word2vec local / word_embeddings / elmo_1024_cased_for_MIT_corpus.txt
, o use incrustaciones de Glove y KazumaChar que también se explotan en el rastreador de estado del diálogo TRADE :

  scripts de python3 / get_Glove-KazumaChar_word_embedding_for_a_dataset.py \
          --in_files data / atis-2 / {entrenar, válido, prueba} \
          --output_word2vec local / word_embeddings / glove-kazumachar_400_cased_for_atis.txt
  scripts de python3 / get_Glove-KazumaChar_word_embedding_for_a_dataset.py \
          --in_files data / snips / {entrenar, válido, prueba} \
          --output_word2vec local / word_embeddings / glove-kazumachar_400_cased_for_snips.txt
  scripts de python3 / get_Glove-KazumaChar_word_embedding_for_a_dataset.py \
          --in_files data / multilingual_task_oriented_data / es / {tren, válido, prueba} \
          --output_word2vec local / word_embeddings / glove-kazumachar_400_cased_for_multilingual_en.txt
  scripts de python3 / get_Glove-KazumaChar_word_embedding_for_a_dataset.py \
          --in_files data / MIT_corpus / {movie_eng, movie_trivia10k13, restaurant} / {train, valid, test} \
          --output_word2vec local / word_embeddings / glove-kazumachar_400_cased_for_MIT_corpus.txt
, o use incrustaciones de palabras en el modelo BERT previamente entrenado:

  scripts de python3 / get_BERT_word_embedding_for_a_dataset.py \
          --in_files data / multilingual_task_oriented_data / es / {tren, válido, prueba} \
          --output_word2vec local / word_embeddings / bert_768_cased_for_multilingual_es.txt \
          --pretrained_tf_type bert --pretrained_tf_name bert-base-multilingual-cased
  scripts de python3 / get_BERT_word_embedding_for_a_dataset.py \
          --in_files data / multilingual_task_oriented_data / es / {tren, válido, prueba} \
          --output_word2vec local / word_embeddings / bert_768_cased_for_multilingual_es.txt \
          --pretrained_tf_type bert --pretrained_tf_name bert-base-multilingual-cased
Ejecute guiones de entrenamiento y evaluación en cada época.
Modelo BLSTM ( slot_tagger )
Modelo BLSTM-CRF ( slot_tagger_with_crf )
Modelo de enfoque Enc-dec (BLSTM-LSTM) ( slot_tagger_with_focus ), el mismo que Encoder-Decoder NN (con entradas alineadas) (Liu y Lane, 2016)
slot_intent_model = slot_tagger # slot_tagger, slot_tagger_with_crf, slot_tagger_with_focus 
fiesta de marcha / atis_with_pretrained_word_embeddings.sh --task_slot_filling $ {} slot_intent_model 
plazo bash / snips_with_pretrained_word_embeddings.sh --task_slot_filling $ {} slot_intent_model 
plazo bash / MIT_corpus_with_pretrained_word_embeddings.sh --task_slot_filling $ {} slot_intent_model - dataroot data / MIT_corpus / movie_eng --dataset mit_movie_eng
bash run / multilingual_en_with_pretrained_word_embeddings.sh --task_slot_filling $ {slot_intent_model} 
bash run / multilingual_es_with_pretrained_word_embeddings.sh --task_slot_filling $ {slot_intent_model} 
bash run / multilingual_intent_th_support_model } bash run / multilingual_embeddings_es_with_pretrained_word_embeddings.sh --task_slot_filling $ {slot_intent_model} bash run / multilingual_intent_th_es_with_pretrained_word_embeddings .
Tutoriales B: Llenado de ranuras y detección de intención con ELMo
Ejecute guiones de entrenamiento y evaluación en cada época.
Modelos ELMo + BLSTM / BLSTM-CRF / Enc-dec focus model (BLSTM-LSTM):
slot_intent_model = slot_tagger # slot_tagger, slot_tagger_with_crf, slot_tagger_with_focus 
fiesta de marcha / atis_with_elmo.sh --task_slot_filling $ {} slot_intent_model 
plazo bash / snips_with_elmo.sh --task_slot_filling   $ {} slot_intent_model 
plazo bash / MIT_corpus_with_elmo.sh --task_slot_filling   $ {} slot_intent_model - dataroot data / MIT_corpus / movie_eng --dataset mit_movie_eng
Tutoriales C: Llenado de ranuras y detección de intención con BERT
Arquitecturas modelo:
BERT conjunto o "con BERT puro ":
bert_SLU_simple

Nuestro BERT + BLSTM (BLSTM-CRF \ Enc-dec focus):
bert_SLU_complex

Ejecute guiones de entrenamiento y evaluación en cada época.
Modelo Pure BERT (sin o con crf):
slot_model = NN # NN, NN_crf 
intent_input = CLS # none, CLS, max, CLS_max 
bash run / atis_with_pure_bert.sh --task_slot_filling $ {slot_model} --task_intent_detection $ {intent_input} 
bash run / snips_sh_ftask_s.bert } --task_intent_detection $ {intent_input} 
bash run / MIT_corpus_with_pure_bert.sh --task_slot_filling $ {slot_model} --task_intent_detection $ {intent_input} --dataroot data / MIT_corpus / movie_eng --dataset mit_movie_eng
fiesta de marcha / multilingual_en_with_pure_bert.sh --task_slot_filling $ {} slot_model --task_intent_detection $ {} intent_input 
plazo bash / multilingual_es_with_pure_bert.sh --task_slot_filling $ {} slot_model --task_intent_detection $ {} intent_input 
fiesta de marcha / multilingual_th_with_pure_bert.sh --task_slot_filling $ {} slot_model --task_intent_detection $ {} intent_input 
fiesta de marcha / ECSA_with_pure_bert.sh --task_slot_filling $ {} slot_model --task_intent_detection $ {} intent_input 
fiesta de marcha / CoNLL2003_NER_with_pure_bert.sh --task_slot_filling $ {} slot_model --task_intent_detection $ {} intent_input
Modelos BERT + BLSTM / BLSTM-CRF / Enc-dec focus model (BLSTM-LSTM):
slot_intent_model = slot_tagger # slot_tagger, slot_tagger_with_crf, slot_tagger_with_focus 
fiesta de marcha / atis_with_bert.sh --task_slot_filling $ {} slot_intent_model 
plazo bash / snips_with_bert.sh --task_slot_filling $ {} slot_intent_model 
plazo bash / MIT_corpus_with_bert.sh --task_slot_filling $ {} slot_intent_model - dataroot data / MIT_corpus / movie_eng --dataset mit_movie_eng
Para optimizar, puede probar BertAdam y AdamW . En mis experimentos, elijo usar BertAdam.
Tutoriales D: Llenado de ranuras y detección de intención con XLNET
Ejecute guiones de entrenamiento y evaluación en cada época.
Modelo Pure XLNET (sin o con crf):
slot_model = NN # NN, NN_crf 
intent_input = CLS # none, CLS, max, CLS_max 
bash run / atis_with_pure_xlnet.sh --task_slot_filling $ {slot_model} --task_intent_detection $ {intent_input} 
bash run / snipsxl_net.spot_pure_ } --task_intent_detection $ {intent_input} 
bash run / MIT_corpus_with_pure_xlnet.sh --task_slot_filling $ {slot_model} --task_intent_detection $ {intent_input} --dataroot data / MIT_corpus / movie_eng --dataset mit_movie_eng
Modelos XLNET + BLSTM / BLSTM-CRF / Enc-dec focus model (BLSTM-LSTM):
slot_intent_model = slot_tagger # slot_tagger, slot_tagger_with_crf, slot_tagger_with_focus 
fiesta de marcha / atis_with_xlnet.sh --task_slot_filling $ {} slot_intent_model 
plazo bash / snips_with_xlnet.sh --task_slot_filling $ {} slot_intent_model 
plazo bash / MIT_corpus_with_xlnet.sh --task_slot_filling $ {} slot_intent_model - dataroot data / MIT_corpus / movie_eng --dataset mit_movie_eng
Para optimizar, puede probar BertAdam y AdamW.
Resultados:
Para los modelos "NLU + BERT / XLNET", los hiperparámetros no se ajustan con cuidado.
Resultados de ATIS:

modelos	intención Acc (%)	ranura F1-puntuación (%)
[Atten. enc-dec NN con entradas alineadas] (Liu y Lane, 2016)	98,43	95,87
[Atten.-BiRNN] (Liu y Lane, 2016)	98.21	95,98
[Enfoque enc-dec] (Zhu y Yu, 2017)	-	95,79
[Slot-Gated] (Goo et al., 2018)	94,1	95,2
Intent Gating y auto-atención	98,77	96,52
BLSTM-CRF + ELMo	97,42	95,62
BERT conjunto	97,5	96,1
Conjunto BERT + CRF	97,9	96,0
BLSTM (A. Emb. De palabras de preentrenamiento de ELMo)	98.10	95,67
BLSTM-CRF (A. Emb. De palabra de preentrenamiento de ELMo)	98,54	95,39
Enfoque enc-dec (A. Emb. De palabras de preentrenamiento de ELMo)	98,43	95,78
BLSTM (A. Emb. De palabras de preentrenamiento de Glove & KazumaChar)	98,66	95,55
BLSTM-CRF (A. Emb. De palabra de preentrenamiento de Glove & KazumaChar)	98.21	95,74
Enfoque enc-dec (A. Emb. De palabras de preentrenamiento de Glove y KazumaChar)	98,66	95,86
BLSTM (B. + ELMo)	98,66	95,52
BLSTM-CRF (B. + ELMo)	98,32	95,62
Enfoque enc-dec (B. + ELMo)	98,66	95,70
BLSTM (C. + BERT)	99,10	95,94
BLSTM (D. + XLNET)	98,77	96.08
Resultados de SNIPS:

El modelo base BERT con carcasa ofrece mejores resultados que el modelo sin carcasa.

modelos	intención Acc (%)	ranura F1-puntuación (%)
[Slot-Gated] (Goo et al., 2018)	97,0	88,8
BLSTM-CRF + ELMo	99,29	93,90
BERT conjunto	98,6	97,0
Conjunto BERT + CRF	98,4	96,7
BLSTM (A. Emb. De palabras de preentrenamiento de ELMo)	99,14	95,75
BLSTM-CRF (A. Emb. De palabra de preentrenamiento de ELMo)	99,00	96,92
Enfoque enc-dec (A. Emb. De palabras de preentrenamiento de ELMo)	98,71	96,22
BLSTM (A. Emb. De palabras de preentrenamiento de Glove & KazumaChar)	99,14	96,24
BLSTM-CRF (A. Emb. De palabra de preentrenamiento de Glove & KazumaChar)	98,86	96,31
Enfoque enc-dec (A. Emb. De palabras de preentrenamiento de Glove y KazumaChar)	98,43	96.06
BLSTM (B. + ELMo)	98,71	96,32
BLSTM-CRF (B. + ELMo)	98,57	96,61
Enfoque enc-dec (B. + ELMo)	99,14	96,69
BLSTM (C. + BERT)	98,86	96,92
BLSTM-CRF (C. + BERT)	98,86	97,00
Enfoque enc-dec (C. + BERT)	98,71	97,17
BLSTM (D. + XLNET)	98,86	97.05
Resultados del conjunto de datos multilingüe de Facebook (nota: el modelo BERT en caja da mejores resultados que el modelo sin caja):

Inglés (en):
modelos	intención Acc (%)	ranura F1-puntuación (%)
Transferencia entre idiomas (solo destino)	99.11	94,81
BLSTM (sin palabra de preentrenamiento emb.)	99,19	95,37
Enfoque enc-dec (A. Emb. De palabras de preentrenamiento de Glove y KazumaChar)	99,28	96.04
BERT puro	99,34	96,23
Español (es):
modelos	intención Acc (%)	ranura F1-puntuación (%)
Transferencia entre idiomas (solo destino)	97,26	80,95
Transferencia entre idiomas (Cross-lingual + ELMo)	97,51	83,38
BLSTM (sin palabra de preentrenamiento emb.) (Solo objetivo)	97,63	86.05
Enfoque enc-dec (A. Emb. De palabra de preentrenamiento de la capa de entrada BERT) (solo destino)	97,67	88,67
Pure BERT (único objetivo)	98,85	89,26
Tailandés (th): (parece que "Enc-dec focus" da mejores resultados que BLSTM)
modelos	intención Acc (%)	ranura F1-puntuación (%)
Transferencia entre idiomas (solo destino)	95,13	87,26
Transferencia entre idiomas (Cross-lingual + Mult. CoVe + auto)	96,87	91,51
BLSTM (sin palabra de preentrenamiento emb.) (Solo objetivo)	96,99	89.17
Enfoque enc-dec (sin preentrenamiento de palabras emb.) (Solo destino)	96,75	91,31
Enfoque enc-dec (A. Emb. De palabra de preentrenamiento de la capa de entrada BERT) (solo destino)	96,87	91.05
Pure BERT (único objetivo)	97,34	92,51
Ranuras F1-puntuaciones de MIT_Restaurant_Movie_corpus (sin intención):

modelos	Restaurante	Movie_eng	Movie_trivia10k13
Dom-Gen-Adv	74.25	83.03	63,51
Joint Dom Spec y Gen-Adv	74,47	85,33	65,33
Aumento de datos mediante la generación de variaciones conjuntas	73,0	82,9	65,7
BLSTM (A. Emb. De palabras de preentrenamiento de ELMo)	77,54	85,37	67,97
BLSTM-CRF (A. Emb. De palabra de preentrenamiento de ELMo)	79,77	87,36	71,83
Enfoque enc-dec (A. Emb. De palabras de preentrenamiento de ELMo)	78,77	86,68	70,85
BLSTM (A. Emb. De palabras de preentrenamiento de Glove & KazumaChar)	78.02	86,33	68,55
BLSTM-CRF (A. Emb. De palabra de preentrenamiento de Glove & KazumaChar)	79,84	87,61	71,90
Enfoque enc-dec (A. Emb. De palabras de preentrenamiento de Glove y KazumaChar)	79,98	86,82	71,10
Ranuras F1-puntuaciones del Asistente de compras de comercio electrónico (ECSA) de Alibaba (sin intención, en chino):

modelos	ranura F1-puntuación (%)
BiLSTM-CRF básico	43.02
BERT puro	46,96
BERT-CRF puro	47,75
Puntuaciones F1 de la entidad de CoNLL-2003 NER (sin intención):

modelos	Puntuación F1 (%)
BERT puro	91,36
BERT-CRF puro	91,55
Modo de inferencia
Un ejemplo aquí:

bash run/atis_with_pretrained_word_embeddings_for_inference_mode__an_example.sh
Referencia
Su Zhu y Kai Yu, "Codificador-decodificador con mecanismo de enfoque para la comprensión del lenguaje hablado basado en el etiquetado de secuencias", en la Conferencia Internacional IEEE sobre Acústica, Habla y Procesamiento de Señales (ICASSP), 2017, págs. 5675-5679.
