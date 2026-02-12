import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from globals import FGSM, PGD, ALPHA, EPSILON, NUM_ITER

def denormalize(batch, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    device = batch.device
    if isinstance(batch, np.ndarray):
        batch = torch.tensor(batch).to(device)
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)
    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


def fgsm_attack(image, data_grad, epsilon = 0.25):
    # Get the sign of the data gradient (element-wise)
    # Create the perturbed image, scaled by epsilon
    # Make sure values stay within valid range
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return perturbed_image


    
def fgsm_loss(model, criterion, inputs, labels, defense_args, return_preds = True):
    alpha = defense_args[ALPHA]
    epsilon = defense_args[EPSILON]

    device = inputs.device

    adv_inputs = inputs.detach().clone().to(device)
    adv_inputs.requires_grad = True
    # Implement the FGSM attack
    # Calculate the loss for the original image
    adv_outputs_for_grad = model(adv_inputs)
    loss_for_grad = criterion(adv_outputs_for_grad, labels)

    model.zero_grad()
    loss_for_grad.backward()

    # Calculate the perturbation
    data_grad = adv_inputs.grad.detach()
    adv_inputs = fgsm_attack(adv_inputs, data_grad, epsilon=epsilon).detach()

    # Calculate the loss for the perturbed image
    # Combine the two losses
    original_outputs = model(inputs)
    adv_outputs = model(adv_inputs)

    
    # Hint: the inputs are used in two different forward passes,
    loss_clean = criterion(original_outputs, labels)
    loss_adv = criterion(adv_outputs, labels)
    loss = alpha * loss_clean + (1.0 - alpha) * loss_adv

    if return_preds:
        _, preds = torch.max(original_outputs, 1)
        return loss, preds
    else:
        return loss


def pgd_attack(model, data, target, criterion, args):
    alpha = args[ALPHA]
    epsilon = args[EPSILON]
    num_iter = args[NUM_ITER]

    # Implement the PGD attack
    # Start with a copy of the data
    original_data = data.detach()
    perturbed_data = original_data.clone()

    # Then iteratively perturb the data in the direction of the gradient
    # Make sure to clamp the perturbation to the epsilon ball around the original data
    # Hint: to make sure to each time get a new detached copy of the data,
    # to avoid accumulating gradients from previous iterations
    # Hint: it can be useful to use toch.nograd()
    for _ in range(num_iter):
        perturbed_data.requires_grad = True

        output = model(perturbed_data)
        loss = criterion(output, target)

        model.zero_grad()
        if perturbed_data.grad is not None:
            perturbed_data.grad.zero_()
        loss.backward()

        with torch.no_grad():
            # 1 step in the direction of grad
            grad_sign = perturbed_data.grad.sign()
            perturbed_data = perturbed_data + alpha * grad_sign

            # Epsilon-ball around original_data
            delta = torch.clamp(perturbed_data - original_data, min=-epsilon, max=epsilon)
            perturbed_data = original_data + delta

            # Clamp to valid pixel range
            perturbed_data = torch.clamp(perturbed_data, 0, 1)

        perturbed_data = perturbed_data.detach()

    return perturbed_data


def test_attack(model, test_loader, attack_function, attack_args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct = 0
    criterion = nn.CrossEntropyLoss()
    adv_examples = []

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True # Very important for attack!
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] 

        # If the initial prediction is wrong, don't attack
        if init_pred.item() != target.item():
            continue

        loss = F.nll_loss(output, target)
        model.zero_grad()
        
        if attack_function == FGSM: 
            # Get the correct gradients wrt the data
            # Perturb the data using the FGSM attack
            # Re-classify the perturbed image
            loss.backward()
            data_grad = data.grad.data

            epsilon = attack_args[EPSILON]
            perturbed_data = fgsm_attack(data, data_grad, epsilon=epsilon)

            output = model(perturbed_data)

        elif attack_function == PGD:
            # Get the perturbed data using the PGD attack
            # Re-classify the perturbed image
            perturbed_data = pgd_attack(model, data, target, criterion, attack_args)
            output = model(perturbed_data)
        else:
            print(f"Unknown attack {attack_function}")

            perturbed_data = data

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] 
        if final_pred.item() == target.item():
            correct += 1
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                original_data = data.squeeze().detach().cpu()
                adv_ex = perturbed_data.squeeze().detach().cpu()
                adv_examples.append( (init_pred.item(), 
                                      final_pred.item(),
                                      denormalize(original_data), 
                                      denormalize(adv_ex)) )

    # Calculate final accuracy
    final_acc = correct/float(len(test_loader))
    print(f"Attack {attack_function}, args: {attack_args}\nTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")
    return final_acc, adv_examples
