import itertools
import json
import torch
import tqdm
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
from utils import \
    get_data_paths, GROUP3, \
    load_img


def load_model() -> SamPredictor:
    sam = sam_model_registry["vit_b"](checkpoint="./sam_vit_b_01ec64.pth")
    predictor = SamPredictor(sam)
    return predictor

def find_norm():
    data_path, mask_path = get_data_paths(GROUP3)
    model = load_model()
    img = load_img(data_path[0])
    # load_file_npz(mask_path[0])
    mask_input_size = [4*x for x in model.model.prompt_encoder.image_embedding_size]
    model.set_image(img)

    mean_candidate = np.linspace(0, 20, 20, dtype=np.float32)
    std = np.linspace(0, 4, 20, dtype=np.float32)
    # Norm dist
    norm_pseudo_mask = torch.randn(1, *mask_input_size, dtype=torch.float32)

    score = []
    N = mean_candidate.shape[0] * std.shape[0]
    for m, s in tqdm.tqdm(
                        itertools.product(mean_candidate, std), 
                        desc='Predicting...', 
                        total=N
                        ):
        _mask = (norm_pseudo_mask + m) * s
        predict_masks, _, _ = model.predict(
            multimask_output=True,
            mask_input=_mask,
            return_logits=False,
        )

        activation_score = predict_masks.sum().astype(np.float32)
        score.append([m, s, activation_score])

    score = np.array(score, dtype=np.float32).tolist()
    
    with open('store.json', 'w') as out:
        json.dump(score, out)



    
    pass

if __name__ == "__main__":
    find_norm()
    import matplotlib.pyplot as plt  # noqa: F401
    import seaborn as sns  # noqa: F401
    import pandas as pd
    with open('store.json') as out:
        metadata = json.load(out)


    df = pd.DataFrame(metadata, columns=['mean', 'std', 'activation'])
    df = df[df['std'] > 0]
    max_activation = df['activation'].max()
    min_activation = df['activation'].min()
    # norm
    df['activation'] = (df['activation'] - min_activation) / (
        max_activation - min_activation)

    sns.displot(df, x='mean', y='std')
    sns.scatterplot(x=df['mean'], y=df['activation'])
    print(metadata)


    # mean_candidate = np.linspace(0, 200, 20, dtype=np.int32)
    # std = np.linspace(0, 10, 20, dtype=np.int32)

    # data = np.zeros((200, 10))
    # dist = [data.min(), data.max(), data.mean(), data.std()]
    
    # for score in metadata:
    #     try:
    #         data[int(score[0]), int(score[1])] = score[2]
    #     except:
    #         pass

    
    # print(dist)
    # plt.imshow(data)
    # plt.show()

    # print(metadata)
    pass