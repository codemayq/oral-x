from mil.config import Config

if __name__ == "__main__":
    if Config.MULTI_LABEL:
        from mil.train import train
    else:
        from mil.train_single import train

    train()