python step2_train_model.py --gpu 1 --time_horizon 5 --seq_len 4 --batch_size 32 --dataset 'cholec80' --exp_name 'cholec80_t=5_s=4'
python step2_train_model.py --gpu 1 --time_horizon 5 --seq_len 10 --batch_size 16 --dataset 'cholec80' --exp_name 'cholec80_t=5_s=10'
python step2_train_model.py --gpu 1 --time_horizon 5 --seq_len 30 --batch_size 8 --dataset 'cholec80' --exp_name 'cholec80_t=5_s=30'
