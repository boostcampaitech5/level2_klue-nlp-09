import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping

import wandb
from pytorch_lightning.loggers import WandbLogger


from pl_train import Model, Dataloader, CustomModelCheckpoint

from utils import seed_everything, load_yaml


if __name__ == "__main__":
    model_dict = {0: "klue_bert_base", 1: "klue_roberta_large", 2: "snunlp_kr_electra", 3: "xlm_roberta_large"}
    model_name = model_dict[1]
    args = load_yaml(model_name)

    # HP Tuning
    # Sweep을 통해 실행될 학습 코드 작성
    sweep_config = {
        "method": "random",  # random: 임의의 값의 parameter 세트를 선택
        "parameters": {
            "learning_rate": {"values": [1e-5, 8e-6, 6e-6, 5e-6]},
            "max_epoch": {"values": [1]},
            "batch_size": {"values": [8, 16]},
            "model_name": {
                "values": [
                    args.model_name,
                    # "klue/roberta-large",
                    # 'monologg/koelectra-base-v3-discriminator',
                    # 'beomi/KcELECTRA-base',
                    # 'rurupang/roberta-base-finetuned-sts',
                    # 'snunlp/KR-ELECTRA-discriminator'
                ]
            },
            # 'warm_up_ratio':{
            #     'values':[0.3, 0.45, 0.6]
            # },
            "weight_decay": {"values": [0, 0.01]},
            # "loss_func": {"values": ["L1"]},
        },
        "metric": {"name": "val_f1", "goal": "maximize"},
    }

    # set version to save model
    def set_version():
        for i in range(1, 1000):
            yield i

    ver = set_version()

    def sweep_train(config=None):
        """
        sweep에서 config로 run
        wandb에 로깅

        Args:
            config (_type_, optional): _description_. Defaults to None.
        """

        with wandb.init(config=config) as run:
            config = wandb.config
            # set seed
            seed_everything(args.seed)
            run.name = f"{config.learning_rate}_{config.batch_size}_{config.weight_decay}_{config.max_epoch}"

            wandb_logger = WandbLogger(project=args.project_name)
            dataloader = Dataloader(config.model_name, config.batch_size, args.shuffle, args.train_path, args.dev_path, args.dev_path, args.predict_path)
            model = Model(
                config.model_name,
                config.learning_rate,
                config.weight_decay,
                args.warmup_steps,
            )

            trainer = pl.Trainer(
                precision="16-mixed",  # 16-bit mixed precision
                accelerator="gpu",  # GPU 사용
                # dataloader를 매 epoch마다 reload해서 resampling
                # reload_dataloaders_every_n_epochs=1,
                max_epochs=config.max_epoch,  # 최대 epoch 수
                logger=wandb_logger,  # wandb logger 사용
                log_every_n_steps=1,  # 1 step마다 로그 기록
                val_check_interval=0.5,  # 0.25 epoch마다 validation
                check_val_every_n_epoch=1,  # val_check_interval의 기준이 되는 epoch 수
                callbacks=[
                    # learning rate를 매 step마다 기록
                    LearningRateMonitor(logging_interval="step"),
                    EarlyStopping("val_f1", patience=5, mode="max", check_finite=False),  # validation f1이 5번 이상 개선되지 않으면 학습을 종료
                    CustomModelCheckpoint(  # validation f1이 기준보다 높으면 저장
                        "./results/", f"klue_rl_{next(ver):0>4}_{{val_f1:.4f}}", monitor="val_f1", save_top_k=1, mode="max"
                    ),
                ],
            )
            trainer.fit(model=model, datamodule=dataloader)  # 모델 학습
            trainer.test(model=model, datamodule=dataloader)  # 모델 평가

    # Sweep 생성
    sweep_id = wandb.sweep(sweep=sweep_config, project=args.project_name)  # config 딕셔너리 추가  # project의 이름 추가
    wandb.agent(sweep_id=sweep_id, function=sweep_train, count=2)  # sweep의 정보를 입력  # train이라는 모델을 학습하는 코드를  # 총 n회 실행
