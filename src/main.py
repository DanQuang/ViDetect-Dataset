import yaml
import argparse
from typing import Text

from task.train import LLM_Detec_Gen_Task
from task.infer import Predict

def main(config_path: Text) -> None:    
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    
    print("Training started...")
    LLM_Detec_Gen_Task(config).training()
    print("Training complete")
    
    print('now evaluate on test data...')
    Predict(config).predict_submission()
    print('task done!!!')
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    
    main(args.config)