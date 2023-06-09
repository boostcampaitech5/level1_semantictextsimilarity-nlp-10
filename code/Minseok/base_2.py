import os
import argparse

import pandas as pd

from tqdm.auto import tqdm

import torch
import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

import transformers
from transformers import PreTrainedTokenizerFast

import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs)


class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path, is_test=False):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, max_length=160)

        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']

        self.automatic_optimization = False

    # 추가 정의 함수 구간.
    def aug_switched_sentence(self, df, switched_columns, frac_v=0.8):
        sampled_train_data = df.sample(
            frac=frac_v, random_state=42, replace=False)
        sampled_train_data[switched_columns[0]], sampled_train_data[switched_columns[1]
                                                                    ] = sampled_train_data[switched_columns[1]], sampled_train_data[switched_columns[0]]
        df = pd.concat([df, sampled_train_data], axis=0)
        df = df.reset_index(drop=True)

        return df
    # 추가 정의 함수 구간.

    def tokenizing(self, df_input):
        data_input = []

        for idx, item in tqdm(df_input.iterrows(), desc='tokenizing', total=len(df_input)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = '[SEP]'.join([item[text_column]
                                 for text_column in self.text_columns])
            # outputs = self.tokenizer.encode(text, return_tensors='pt', add_special_tokens=True, padding='max_length', truncation=True, max_length=160)
            outputs = self.tokenizer(
                text, add_special_tokens=True, padding='max_length', truncation=True, max_length=160)
            data_input.append(outputs['input_ids'])

        return data_input

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            # new_train_data = pd.read_csv("new_dataset.csv")
            # train_data = pd.concat([train_data, new_train_data])
            val_data = pd.read_csv(self.dev_path)

            # 학습데이터 준비
            train_data = self.aug_switched_sentence(
                train_data, switched_columns=self.text_columns)
            # 다양한 data aug는 여기에서
            self.after_aug_train_data = train_data
            train_inputs, train_targets = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        def make_sampler(train_data):
            train_data['class'] = train_data['label'].apply(
                lambda x: int(x//1) if x != 5 else int(4))
            class_counts = train_data['class'].value_counts().to_list()
            # [3711, 1715, 1393, 1368, 1137]
            labels = train_data['class'].to_list()
            # [2, 4, 2, 3, 0, 2, 3, 0,
            num_samples = sum(class_counts)
            class_weights = [num_samples/class_counts[i]
                             for i in range(len(class_counts))]
            # [2.51, 5.436, 6.69, 6.81, 8.20]
            weights = [class_weights[labels[i]]
                       for i in range(int(num_samples))]
            # [6.69, 8.20, 6.69, 6.81, 2.51, 6.69
            sampler = torch.utils.data.WeightedRandomSampler(
                torch.DoubleTensor(weights), int(num_samples))
            return sampler

        # shuffle=args.shuffle
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8, sampler=make_sampler(self.after_aug_train_data))

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)


class Model(pl.LightningModule):
    def __init__(self, model_name, lr, warmup_step):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        self.warmup_step = warmup_step

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1)

        # Loss 계산을 위해 사용될 L2Loss로 변경.
        self.loss_func = torch.nn.MSELoss()

    def forward(self, x):
        x = self.plm(x)['logits']
        # x = torch.sum(x, 1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(
            logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(
            logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, total_iters=self.warmup_step)

        return (
            [optimizer],
            [
                {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                    'reduce_on_plateau': False,
                    'monitor': 'val_loss',
                }
            ]
        )


if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=True)
    parser.add_argument('--project_name', default='sts')
    parser.add_argument('--entity_name', default='nlp-10')
    parser.add_argument('--sweeps_cnt', default=5)
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

    # args.train = True if args.train == "True" else False
    # args.continue_train = True if args.continue_train == "True" else False
    print(args)

    # sweep_config 생성.
    sweep_config = {'name': f"{args.model_name}_based-wm-step_bs_ep-5",  # name : sweep_name
                    'method': 'grid',  # 'grid', 'uniform', 'bayesian'
                    'parameters': {
                        'lr': {  # parameter  작성방식 여러개 있으니까, 노션 문서 참고
                            'values': [1e-5]
                        },
                        'warmup_step': {
                            "values": [100]
                        },
                        "batch_size": {
                            "values": [32]
                        },
                        "max_epoch": {
                            "values": [5]
                        },
                    },
                    # goal : maximize, minimize
                    'metric': {'name': 'val_pearson', 'goal': 'maximize'}
                    }
    # sweep_config 생성.

    # Wandb logging 설정.
    def sweep_train(config=None):
        run = wandb.init(config=config)
        config = wandb.config
        # run_name : sweep 안에 학습을 시키고있는 학습환경의 이름
        run.name = f"model: {args.model_name} / batch_size: {config.batch_size} / lr: {config.lr} / warmup: {config.warmup_step}"

        # seed_everything import 하시고 사용하셔야 합니다. [ 모델 생성 및 data loader 학습 코드 위치 ]
        seed_everything(10, workers=True)

        dataloader = Dataloader(args.model_name, config.batch_size, args.shuffle, args.train_path, args.dev_path,
                                args.test_path, args.predict_path)

        # dataloader와 model을 생성합니다.
        model = Model(args.model_name, config.lr, config.warmup_step)
        wandb_logger = WandbLogger()

        # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
        trainer = pl.Trainer(accelerator='gpu', max_epochs=config.max_epoch,
                             log_every_n_steps=1, logger=wandb_logger)

        # Train part
        trainer.fit(model=model, datamodule=dataloader)

        trainer.test(model=model, datamodule=dataloader)

        # 학습이 완료된 모델을 저장합니다.
        torch.save(model, f'model_{run.id}.pt')
        # sweep 한번 finish 시에, slack 알림 발송입니다.
        run.alert(title="Training Finshed",
                  text=f"[batch_size : {config.batch_size} / lr : {config.lr}] Model Finished", level=wandb.AlertLevel.INFO)
        # run.finish()
    # Wandb logging 설정.

    # args.train = False

    if args.train:
        print("Train mode")
        sweep_id = wandb.sweep(
            sweep_config, project=args.project_name, entity=args.entity_name)
        # count : sweep을 실행할 횟수
        wandb.agent(sweep_id, sweep_train, count=args.sweeps_cnt)
        wandb.finish()

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
        model = torch.load('model_bat_32_lr_1e-05_wm-100.pt')
        predictions = trainer.predict(model=model, datamodule=dataloader)

        # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
        predictions = list(round(float(i), 5) for i in torch.cat(predictions))

        # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
        output = pd.read_csv('./data/sample_submission.csv')
        output['target'] = predictions
        output.to_csv('output.csv', index=False)
