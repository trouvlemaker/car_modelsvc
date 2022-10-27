modelname=damage_cascade-faster-rcnn_gn_anchor1_1x

CUDA_VISIBLE_DEVICES=1 python ../custom_predict.py \
--predict damage_bbox/eval \
--load train_log/${modelname}-20190905/checkpoint \
--date=20190909 \
--output_dir=pred_results/${modelname}_resout \
--num_output_images=4000 \
--config=../configs/$modelname.json \
--evaluate=pred_results/${modelname}_res.json
