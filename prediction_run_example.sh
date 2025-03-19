docker run -it\
    --mount type=bind,src=./data,dst=/root/trypanodeepscreen/data\
    --mount type=bind,src=./trained_models,dst=/root/trypanodeepscreen/trained_models\
    --mount type=bind,src=./config,dst=/root/trypanodeepscreen/config\
    --mount type=bind,src=./predictions,dst=/root/trypanodeepscreen/predictions\
    --shm-size=15gb\
    trypanodeepscreen \
        predict \
            --model_folder_path trained_models/trypano_experiment_example \
            --data_input_prediction data/predict_data_example.csv \
            --result_path_prediction_csv predictions/prediction_results_example.csv \
            --metric_ensambled_prediction val_auroc \
            --n_checkpoints 26
