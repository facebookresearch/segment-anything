from torch.optim import Adam
import monai
from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize




def train(model, train_dataloader, val_dataloader, device, epochs=100,multimask_output=False):
    '''
    # Note: Hyperparameter tuning could improve performance here
    '''

    optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    model.to(device)

    model.train()
    for epoch in range(epochs):
        epoch_losses = []
        for batch in tqdm(train_dataloader):
          # forward pass
          outputs = model(pixel_values=batch["pixel_values"].to(device),
                          input_boxes=batch["input_boxes"].to(device),
                          multimask_output=multimask_output)

          # compute loss
          predicted_masks = outputs.pred_masks.squeeze(1)
          ground_truth_masks = batch["ground_truth_mask"].float().to(device)
          loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

          # backward pass (compute gradients of parameters w.r.t. loss)
          optimizer.zero_grad()
          loss.backward()

          # optimize
          optimizer.step()
          epoch_losses.append(loss.item())

        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean(epoch_losses)}')


if __name__ == "__main__":
    print("This is the train.py file.")



