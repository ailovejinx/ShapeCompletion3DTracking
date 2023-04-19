# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
# --nproc_per_node=1 --master_port=9999 main.py \
# --train_model \
# --batch_size=128 \
# --model_name=first_test_dist_single \
# --dataset_path=../data/kitti/training \
# --max_num_worker=32 \

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 main.py \
--train_model \
--batch_size=128 \
--model_name=first_test_dist_4 \
--dataset_path=../data/kitti/training \
--max_num_worker=16