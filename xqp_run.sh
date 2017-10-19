export MXNET_CPU_WORKER_NTHREADS=12
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python xqp_fine-tune.py --pretrained-model model/resnet-152 \
    --load-epoch 0 --gpus 0 \
    --data-train data/scene_train_20170904.lst --model-prefix model/Scence-restnet-152 \
    --data-val data/scence_validation_20170908.lis \
	--data-nthreads 12 \
    --batch-size 64 --num-classes 80 --num-examples 53879
