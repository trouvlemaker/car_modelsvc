date=20190909
model_name=damage_cascade-faster-rcnn_gn_anchor1_1x-retrain

CUDA_VISIBLE_DEVICES='0' python ../custom_train.py \
--logdir=train_log/$model_name-$date \
--date=$date \
--config=../configs/$model_name.json \
--model_name=$model_name \
--load=train_log/damage_cascade-faster-rcnn_gn_anchor1_1x-20190905/checkpoint