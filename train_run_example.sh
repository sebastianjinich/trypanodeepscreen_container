docker run -d\
    --mount type=bind,src=./data,dst=/root/trypanodeepscreen/data\
    --mount type=bind,src=./trained_models,dst=/root/trypanodeepscreen/trained_models\
    --mount type=bind,src=./config,dst=/root/trypanodeepscreen/config\
    --mount type=bind,src=./predictions,dst=/root/trypanodeepscreen/predictions\
    --shm-size=15gb\
    trypanodeepscreen \
        train \
            --data_train_val_test data/train_data_example.csv \
            --target_name trypano_experiment_example \
            --experiment_result_path trained_models/ 
