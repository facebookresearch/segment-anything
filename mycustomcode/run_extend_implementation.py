from segment_anything import sam_model_registry
import torch
import parser
from omegaconf import OmegaConf

device = "cuda" if torch.cuda.is_available() else "cpu"


supported_tasks = ['detection', 'semantic_seg', 'instance_seg']




def getargs():
    args = parser.parse_args()
    task_name = args.task_name
    if args.cfg is not None:
        config = OmegaConf.load(args.cfg)
    else:
        assert task_name in supported_tasks, "Please input the supported task name."
        config = OmegaConf.load("./config/{task_name}.yaml".format(task_name=args.task_name))

    train_cfg = config.train
    val_cfg = config.val
    test_cfg = config.test


def runTrainingBreastCancer(model, optimizer, losses, train_loader, val_loader, scheduler):
    num_epochs = 100

    # ********************* 1. Load the dataset *********************
    # Load the dataset
    dataset = load_dataset("nielsr/breast-cancer", split="train")
    # ********************* 2. Preprocess the dataset *********************
    # Preprocess the dataset
    #processor = get_processor()
    # ********************* 3. Load the model *********************
    sam_model = sam_model_registry['vit_b'](checkpoint=r'./models/sam_vit_b_01ec64.pth')
    sam_model.to(device)
    # ********************* 4. Prepare the model for training *********************
    for name, param in sam_model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)



    train.train(sam_model, None, None, device, epochs=num_epochs)


    pass

if __name__ == "__main__":
    print("This is the file.")

    runTrainingBreastCancer()


