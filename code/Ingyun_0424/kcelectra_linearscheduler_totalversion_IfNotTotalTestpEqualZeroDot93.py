import argparse
from typing import Optional, Union, Sequence, Mapping, Any

import pandas as pd
import transformers
import torch
import torchmetrics
import pytorch_lightning as pl
import wandb
import math
import numpy as np
import re
# !pip install emoji
import emoji

# !pip install soynlp
from soynlp.normalizer import repeat_normalize

from tqdm.auto import tqdm
from CustomScheduler import CosineAnnealingWarmUpRestarts
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.model_selection import KFold, StratifiedKFold


# !pip install git+https://github.com/jungin500/py-hanspell
from hanspell import spell_checker











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
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path, num_split=10, k=1):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path
        self.num_split = num_split
        self.k = k


        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=140)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']
        self.source_columns = 'source'

        self.pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣]+')
        self.url_pattern = re.compile(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

    #KcElectra 사용을 위한 Text Preprocessing
    def clean(self, x):
        x = self.pattern.sub(' ', x)
        x = emoji.replace_emoji(x, replace='') #emoji 삭제
        x = self.url_pattern.sub('', x)
        x = x.strip()
        x = repeat_normalize(x, num_repeats=2)
        return x

    def switched_aug_sentence(self, df, switched_columns, frac_v=0.8):
        sampled_train_data = df.sample(frac=frac_v, random_state=42, replace=False)
        sampled_train_data[switched_columns[0]], sampled_train_data[switched_columns[1]] = sampled_train_data[switched_columns[1]], sampled_train_data[switched_columns[0]]
        df = pd.concat([df, sampled_train_data], axis=0)
        df = df.reset_index(drop=True)
        return df


    def reduce_sampling_label_close_zero(self, df, frac_v=0.6):
        df = df.drop(df[df['label'] < 0.3]['label'].sample(frac=frac_v, random_state = 42, replace=False, axis=0).index)
        return df


    def aug_label_close_five(self, df, frac_v=0.1):
        copy_df = df.sample(frac=frac_v, random_state=42, replace=False)
        copy_df['sentence_2'] = copy_df['sentence_1']
        copy_df['label'] = 5
        df = pd.concat([df, copy_df], axis=0)
        return df
    def prepro_spell_checker(self, data):
        data[self.text_columns] = data[self.text_columns].applymap(
            lambda x : spell_checker.check(re.sub("&", " ", x)).checked)
        return data
    





    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            # Source Data 추가
            text = item[self.source_columns] + '[SEP]'
            text += '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=140)
            data.append(outputs['input_ids'])
        return data

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # KcElectra 사용을 위한 Text Preprocessing
        # data[self.text_columns] = data[self.text_columns].applymap(self.clean)

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
            train_data = pd.read_csv(self.train_path)#수정
            val_data = pd.read_csv(self.dev_path)
            val_data = self.prepro_spell_checker(val_data)

            train_data = pd.concat([train_data, val_data], axis=0)
            train_data = self.reduce_sampling_label_close_zero(train_data)
            train_data = self.aug_label_close_five(train_data)
            train_data = self.switched_aug_sentence(train_data, switched_columns=self.text_columns, frac_v=0.8)
            
            self.after_aug_train_data = train_data
            #수정


            # # total_data = pd.concat([train_data, val_data], axis=0)
            # total_input, total_targets = self.preprocessing(train_data)
            # total_dataset = Dataset(total_input, total_targets)
            #
            # # kf = KFold(n_splits=self.num_split, shuffle=True, random_state=10)
            # kf = StratifiedKFold(n_splits=self.num_split, shuffle=True, random_state=10)
            # all_splits = [k for k in kf.split(total_dataset, train_data['label'].apply(lambda x : np.round(x)))]
            #
            # train_indexes, val_indexes = all_splits[self.k]
            # train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()
            #
            #
            # self.train_dataset = [total_dataset[x] for x in train_indexes]
            # self.val_dataset = [total_dataset[x] for x in val_indexes]

            # 학습데이터 준비
            train_inputs, train_targets = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_data = self.prepro_spell_checker(test_data)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_data = self.prepro_spell_checker(predict_data)
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
        #수정        
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)#, sampler=make_sampler(self.after_aug_train_data))

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)
    











class Model(pl.LightningModule):
    def __init__(self, model_name, lr, weight_decay=0, warmup_ratio=1, gamma=0.5, start_factor=0.2):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.gamma = gamma
        self.start_factor = start_factor

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1)
        self.plm.classifier.Dropout = torch.nn.Dropout(p=0.5, inplace=False)
        self.loss_func = torch.nn.MSELoss()

    def forward(self, x):
        x = self.plm(x)['logits']

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(torch.minimum(torch.maximum(logits, torch.zeros_like(logits)), torch.full_like(logits, 5)), y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(torch.minimum(torch.maximum(logits, torch.zeros_like(logits)), torch.full_like(logits, 5)), y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        logits = torch.minimum(torch.maximum(logits, torch.zeros_like(logits)), torch.full_like(logits, 5))

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)
        logits = torch.minimum(torch.maximum(logits, torch.zeros_like(logits)), torch.full_like(logits, 5))

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, eps=1e-8)

        # General Warmup Scheduler
        return {
            'optimizer': optimizer,
            'lr_scheduler' : torch.optim.lr_scheduler.LinearLR(optimizer,  start_factor=self.start_factor,
                                end_factor=1.0, total_iters=self.warmup_ratio),
            # 'lr_scheduler' : torch.optim.lr_scheduler.LinearLR(optimizer,  start_factor=0.3,
            #                     end_factor=1.0, total_iters=math.ceil(self.trainer.estimated_stepping_batches * self.warmup_ratio)),
            # 'lr_scheduler': CosineAnnealingWarmUpRestarts(optimizer,
            #                                               T_0=self.trainer.estimated_stepping_batches//10,
            #                                               T_mult=1, eta_max=self.lr,
            #                                               T_up=math.ceil(self.trainer.estimated_stepping_batches * self.warmup_ratio),
            #                                               gamma=self.gamma, last_epoch=-1),
            'monitor': 'val_loss'
        }

        # Base Test Scheduler
        # return {
        #     'optimizer': optimizer,
        #     'lr_scheduler': torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, step_size_up=5, max_lr=self.lr, step_size_down=5, cycle_momentum=False, mode='triangular2', gamma=0.5, scale_fn=None, scale_mode='cycle', last_epoch=-1, verbose=False),
        #     'monitor': 'val_loss'
        # }

















# 하이퍼 파라미터 등 각종 설정값을 입력받습니다
# 터미널 실행 예시 : python3 run.py --batch_size=64 ...
# 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default="beomi/KcELECTRA-base-v2022", type=str)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--max_epoch', default=3, type=int)
parser.add_argument('--learning_rate', default=5e-5, type=float)
parser.add_argument('--shuffle', default=True)
parser.add_argument('--train_path', default='./data/train.csv')
parser.add_argument('--dev_path', default='./data/dev.csv')
parser.add_argument('--test_path', default='./data/dev.csv')
parser.add_argument('--predict_path', default='./data/test.csv')
args = parser.parse_args(args=[])








sweep_ingyun_config = {'name': f"{args.model_name}-sweep",
                'method': 'grid',
                'parameters': {'lr': {
                                    'values': [3e-5]#######
                                },
                                "batch_size": {
                                    "values": [32]########
                                },
                                "max_epoch": {
                                    "values": [12]#########30분짜리가 36개.
                                },
                                "weight_decay": {
                                    'values' : [0.1]
                                },
                                "warmup_ratio": {
                                    'values' : [4]######### total_iter에 바로 먹일 것
                                },
                                "gamma": {
                                    'values' : [0.3]
                                },
                                "start_factor": {
                                    'values' : [0.28]##########
    
                                }
                                # "num_splits": {
                                #     "values": [5]
                                # }
                            },
                'metric': {'name': 'val_loss', 'goal': 'minimize'}}











def sweep_train(config=None):
    global trainer, model, dataloader, run
    #수정

    # results = []
    seed_everything(42, workers=True)
    run = wandb.init(config=config)
    config = wandb.config
    run.name = f"ingyuntotal/batch{config.batch_size}/epoch{config.max_epoch}/lr{config.lr}/start{config.start_factor}/warm{config.warmup_ratio}"
    # run.name = f"model: {args.model_name} / batch_size: {config.batch_size} / lr: {config.lr} / weight_decay: {config.weight_decay}"

    dataloader = Dataloader(args.model_name, config.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path)
    model = Model(args.model_name, config.lr, config.weight_decay, config.warmup_ratio, config.gamma, config.start_factor)
    wandb_logger = WandbLogger()

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min')
    trainer = pl.Trainer(max_epochs=config.max_epoch, logger=wandb_logger, log_every_n_steps=1, callbacks=[early_stopping], deterministic=False)
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)
    torch.save(model, f"sweep_{run.id}.pt")
    run.alert(title="Training Finshed", text=f"[batch_size : {config.batch_size} / lr : {config.lr}] Model Finished", level=wandb.AlertLevel.INFO)
    

    predictions = trainer.predict(model=model, datamodule=dataloader)
    predictions = list(round(float(i), 5) for i in torch.cat(predictions))
    output = pd.read_csv('./data/sample_submission.csv')
    output['target'] = predictions
    output.to_csv(f'elec_sampler_spacing_total_{run.id}.csv', index=False)



sweep_id = wandb.sweep(sweep_ingyun_config, project="sts", entity="nlp-10")
wandb.agent(sweep_id, sweep_train)#, count=4)
wandb.finish()








# # dataloader = Dataloader(args.model_name, 8, args.shuffle, args.train_path, args.dev_path,
# #                         args.test_path, args.predict_path)
# # model = torch.load("sweep_hqflfrz0.pt")
# # trainer = pl.Trainer(max_epochs=3)




# seed_everything(42)
# dataloader = Dataloader(args.model_name, 수작업batch_size, args.shuffle, args.train_path, args.dev_path,
#                         args.test_path, args.predict_path)
# model = torch.load("sweep_수작업hqflfrz0.pt")
# trainer = pl.Trainer(max_epochs=수작업)#아마 어차피 추론에서는 안쓸듯
# #dataloader, model 수정
# predictions = trainer.predict(model=model, datamodule=dataloader)

# # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
# predictions = list(round(float(i), 5) for i in torch.cat(predictions))
# #수정

# # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
# output = pd.read_csv('./data/sample_submission.csv')
# output['target'] = predictions
# output.to_csv(f'out_elec_수작업.csv', index=False)