import argparse

import pandas as pd

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from base import Model, Dataloader

if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='True')
    parser.add_argument('--project_name', default='')
    # parser.add_argument('--continue_train', default='False')

    parser.add_argument('--model_name', default='klue/roberta-small', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=3, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='./data/train.csv')
    parser.add_argument('--dev_path', default='./data/dev.csv')
    parser.add_argument('--test_path', default='./data/dev.csv')
    parser.add_argument('--predict_path', default='./data/test.csv')
    args = parser.parse_args()

    args.train = True if args.train == "True" else False
    args.continue_train = True if args.continue_train == "True" else False
    print(args)

    if args.train:
        print("Train mode")
        wandb_logger = WandbLogger(project=args.project_name)

        # dataloader와 model을 생성합니다.
        dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                                args.test_path, args.predict_path)
        model = Model(args.model_name, args.learning_rate)

        # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
        trainer = pl.Trainer(accelerator='gpu', max_epochs=args.max_epoch,
                             log_every_n_steps=1, logger=wandb_logger)

        # Train part
        trainer.fit(model=model, datamodule=dataloader)

        dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                                args.test_path, args.predict_path)
        trainer.test(model=model, datamodule=dataloader)

        # 학습이 완료된 모델을 저장합니다.
        torch.save(model, 'model.pt')
    else:
        print("Inference mode")
        # dataloader와 model을 생성합니다.
        dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                                args.test_path, args.predict_path)

        # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
        trainer = pl.Trainer(
            accelerator='gpu', max_epochs=args.max_epoch, log_every_n_steps=1)

        # Inference part
        # 저장된 모델로 예측을 진행합니다.
        model = torch.load('model.pt')
        predictions = trainer.predict(model=model, datamodule=dataloader)

        # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
        predictions = list(round(float(i), 1) for i in torch.cat(predictions))

        # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
        output = pd.read_csv('./data/sample_submission.csv')
        output['target'] = predictions
        output.to_csv('output.csv', index=False)
