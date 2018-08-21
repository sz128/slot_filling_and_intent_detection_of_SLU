# SLU_focus_and_crf
 * An implementation for "focus" part of the paper "Encoder-decoder with focus-mechanism for sequence labelling based spoken language understanding".
 * An implementation of BLSTM-CRF based on [https://github.com/jiesutd/NCRFpp/blob/master/model/crf.py]

## Setup
 * pytorch 0.4.0
 * python 3.6.x
 * pip install gpustat     [if gpu is used]

## Running
 * ./run/run_slot_tagging.sh     [set deviceId=-1 if you are going to use cpu for training]
 * You can get full atis data from https://github.com/yvchen/JointSLU .

## Reference
 * Su Zhu and Kai Yu, "Encoder-decoder with focus-mechanism for sequence labelling based spoken language understanding," in IEEE International Conference on Acoustics, Speech and Signal Processing(ICASSP), 2017, pp. 5675-5679.
