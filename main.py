from src.unimat.dataset import all_in_one
from src.training import train_full_workflow
from src.generate import main as generate_main

if __name__ == "__main__":
    all_in_one()
    # train_full_workflow(ds_path="./data/unimat/torch", vae_epochs=1, sd_epochs=1)
    # generate_main()