cd .. #starting point must be at src/
python main.py --step "make_data" --environment "local"
python main.py --step "feat_data" --environment "local"
python main.py --step "train_model" --environment "local"
python main.py --step "predict" --environment "local"