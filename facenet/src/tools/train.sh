python ../train_softmax.py \
--logs_base_dir ~/logs/facenet/ \
--models_base_dir ~/models/facenet/raw_train \
--data_dir ~/dataset/video-faces/face_cap_cluster_aug_align \
--image_size 160 \
--model_def models.inception_resnet_v1 \
--optimizer ADAM \
--learning_rate -1 \
--max_nrof_epochs 30 \
--epoch_size 790 \
--keep_probability 0.8 \
--random_crop \
--random_flip \
--learning_rate_schedule_file ../../data/learning_rate_schedule_classifier_casia.txt \
--weight_decay 5e-4 \
--embedding_size 512 \


