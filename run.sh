feature_type=("mel" "mfcc" )
output_dir="preprocessed"
for ft in "${feature_type[@]}"; do
    echo "Processing feature type: $ft"
    python data_preprocess.py \
    --root IEMOCAP_full_release \
    --output_dir ${output_dir} \
    --feature_type ${ft} &

done
wait


cuda_idx=2
model_type=("cnn" "nn")
for ft in "${feature_type[@]}"; do
    for mt in "${model_type[@]}"; do
        echo "Training model for feature type: $ft and model type: $mt"
        CUDA_VISIBLE_DEVICES=$cuda_idx python train.py \
        --config configs/${ft}_${mt}.yaml \
        --data-dir ${output_dir}/${ft} \
        --h5 ${output_dir}/${ft}/iemocap_${ft}.h5 \
        --labels-csv ${output_dir}/${ft}/iemocap_labels.csv \
        --device "cuda:0" \
        --save-dir "exp/${ft}_${mt}" &

    done
    cuda_idx=$((cuda_idx + 1))
done

wait