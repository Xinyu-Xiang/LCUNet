CUDA_VISIBLE_DEVICES=0 python train_test_feature_ours_phase2_upload.py --train_dataset "DOLOS" \
    --train_list './dataset2/DOLOS_train_l464_t365.pkl' \
    --test_dataset "MMDD" \
    --test_list './dataset2/MMDD_test_features.pkl' \
    --fusion_type 'concat'\
    --modalities "vaf" \
    --log "logsintra_bs16_ep30_1" \
    --fused_weight 1 \
    --batchsize 16 \
    --epochs 14