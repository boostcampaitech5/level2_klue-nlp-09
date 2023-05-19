import wandb
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pl_train import Model, Dataloader, CustomModelCheckpoint
from utils import seed_everything, load_yaml


if __name__ == "__main__":
    model_dict = {0: "klue_bert_base", 
                  1: "klue_roberta_large", 
                  2: "snunlp_kr_electra", 
                  3: "xlm_roberta_large", 
                  4: "skt_kogpt2",
                  5: "twhin_bert_large"}
    model_name = model_dict[3]
    args = load_yaml(model_name)

    # HP Tuning
    # Sweep을 통해 실행될 학습 코드 작성
    sweep_config = {
        "method": "bayes",  # random: 임의의 값의 parameter 세트를 선택
        "parameters": {
            "learning_rate": {"values": [5e-5, 1e-5, 5e-6, 1e-6, 5e-7]},
            "batch_size": {"values": [8, 16, 24]},
            "dropout": {"values": [0.1, 0.2, 0.3]},
            "warmup_steps": {"values": [None, 500, 1000, 2000, 3000]},
            "weight_decay": {"values": [0, 0.01, 0.03, 0.05, 0.10]},
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
            # set seed
            seed_everything(args.seed)

            config = wandb.config
            ws_nickname = config.warmup_steps if config.warmup_steps else 0
            run.name = f"LR{config.learning_rate}_\
                BS{config.batch_size}_\
                    DO{config.dropout}_\
                        WS{ws_nickname}_\
                            WD{config.weight_decay}"

            wandb_logger = WandbLogger(project=args.project_name)
            dataloader = Dataloader(
                model_name=args.model_name,
                batch_size=config.batch_size,
                shuffle=args.shuffle,
                tem=args.tem,
                train_path=args.train_path,
                dev_path=args.dev_path,
                test_path=args.dev_path,
                predict_path=args.predict_path,
                # args.data_clean,
                # args.data_aug,
            )

            vocab_size = len(dataloader.tokenizer)
            model = Model(
                model_name=args.model_name,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                vocab_size=vocab_size,
                dropout=config.dropout,
                warmup_steps=config.warmup_steps
            )

            trainer = pl.Trainer(
                precision="16-mixed",  # 16-bit mixed precision
                accelerator="gpu",  # GPU 사용
                # dataloader를 매 epoch마다 reload해서 resampling
                # reload_dataloaders_every_n_epochs=1,
                max_epochs=args.num_train_epochs,  # 최대 epoch 수
                logger=wandb_logger,  # wandb logger 사용
                log_every_n_steps=1,  # 1 step마다 로그 기록
                val_check_interval=0.5,  # 0.25 epoch마다 validation
                check_val_every_n_epoch=1,  # val_check_interval의 기준이 되는 epoch 수
                callbacks=[
                    # learning rate를 매 step마다 기록
                    LearningRateMonitor(logging_interval="step"),
                    EarlyStopping("val_loss", patience=6, mode="min", check_finite=False),  # validation f1이 5번 이상 개선되지 않으면 학습을 종료
                    CustomModelCheckpoint(  # validation f1이 기준보다 높으면 저장
                        "./results/", f"{args.model_name}_{next(ver):0>4}_{{val_f1:.4f}}", monitor="val_f1", save_top_k=1, mode="max"
                    ),
                ],
            )
            trainer.fit(model=model, datamodule=dataloader)  # 모델 학습
            trainer.test(model=model, datamodule=dataloader)  # 모델 평가

    # Sweep 생성
    sweep_id = wandb.sweep(sweep=sweep_config, project=args.project_name)  # config 딕셔너리 추가  # project의 이름 추가
    wandb.agent(sweep_id=sweep_id, function=sweep_train, count=15)  # sweep의 정보를 입력  # train이라는 모델을 학습하는 코드를  # 총 n회 실행
