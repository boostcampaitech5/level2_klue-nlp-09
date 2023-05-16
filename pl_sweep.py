import wandb
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pl_train import Model, Dataloader, CustomModelCheckpoint
from utils import seed_everything, load_yaml
import torch
import pandas as pd


if __name__ == "__main__":
    model_dict = {
        0: "klue_bert_base",
        1: "klue_roberta_large",
        2: "snunlp_kr_electra",
        3: "xlm_roberta_large",
        4: "skt_kogpt2",
    }
    model_name = model_dict[1]
    args = load_yaml(model_name)

    # HP Tuning
    # Sweep을 통해 실행될 학습 코드 작성
    sweep_config = {
        "method": "bayes",  # random: 임의의 값의 parameter 세트를 선택
        "parameters": {
            "learning_rate": {"values": [1e-5, 5e-6, 1e-6]},
            "max_epoch": {"values": [10, 15, 20]},
            "batch_size": {"values": [16, 32, 64]},
            "dropout": {"values": [0.0]},
            "tem": {"values": ["punct"]},
            "train_path": {"values": ["~/dataset/train/v1dh/train_95.csv"]},
            "warmup_steps": {"values": [None, 500]},
            "weight_decay": {"values": [0, 0.01, 0.05]},
        },
        "metric": {"name": "val_loss", "goal": "minimize"},
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
            if config.tem == "none":
                te_nickname = "no"
            elif config.tem == "non_punct":
                te_nickname = "np"
            elif config.tem == "punct":
                te_nickname = "pu"
            tp_nickname = (
                "base"
                if config.train_path == "~/dataset/train/train_90_ksh.csv"
                else "aug"
            )
            ws_nickname = config.warmup_steps if config.warmup_steps else 0
            # set seed
            seed_everything(args.seed)
            run.name = f"LR{config.learning_rate}_\
                ME{config.max_epoch}_\
                    BS{config.batch_size}_\
                        DO{config.dropout}_\
                            TE{te_nickname}_\
                                TP{tp_nickname}_\
                                    WS{ws_nickname}_\
                                        WD{config.weight_decay}"

            wandb_logger = WandbLogger(project=args.project_name)
            dataloader = Dataloader(
                model_name=args.model_name,
                batch_size=config.batch_size,
                shuffle=args.shuffle,
                tem=config.tem,
                train_path=config.train_path,
                dev_path=args.dev_path,
                test_path=args.dev_path,
                predict_path=args.predict_path,
                # args.data_clean,
                # args.data_aug,
            )

            def compute_class_weights(train_path):
                train_df = pd.read_csv(train_path)
                class_counts = train_df["label"].value_counts().sort_index()
                total_samples = train_df.shape[0]
                class_weights = total_samples / (
                    len(class_counts) * class_counts.astype(float)
                )
                print(class_weights)
                class_weights = torch.tensor(class_weights).to("cuda:0")
                class_weights = class_weights.float()
                return class_weights

            vocab_size = len(dataloader.tokenizer)
            model = Model(
                model_name=args.model_name,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                vocab_size=vocab_size,
                dropout=config.dropout,
                class_weights=compute_class_weights(args.train_path),
                warmup_steps=config.warmup_steps,
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
                    EarlyStopping(
                        "val_loss", patience=6, mode="min", check_finite=False
                    ),  # validation f1이 5번 이상 개선되지 않으면 학습을 종료
                    CustomModelCheckpoint(  # validation f1이 기준보다 높으면 저장
                        "./results/",
                        f"{args.model_name}_{next(ver):0>4}_{{val_f1:.4f}}",
                        monitor="val_f1",
                        save_top_k=1,
                        mode="max",
                    ),
                ],
            )
            trainer.fit(model=model, datamodule=dataloader)  # 모델 학습
            trainer.test(model=model, datamodule=dataloader)  # 모델 평가

    # Sweep 생성
    sweep_id = wandb.sweep(
        sweep=sweep_config, project=args.project_name
    )  # config 딕셔너리 추가  # project의 이름 추가
    wandb.agent(
        sweep_id=sweep_id, function=sweep_train, count=10
    )  # sweep의 정보를 입력  # train이라는 모델을 학습하는 코드를  # 총 n회 실행
