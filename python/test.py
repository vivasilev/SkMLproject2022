import torch
import numpy as np

from tqdm import tqdm


def test(device, background_dataloader,
         synt_net, gen_net, rend_net, rec_net, n, m):
    errors = torch.tensor([]).to(device)

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for batch in background_dataloader:
            # for i in tqdm(range(int(num_strings / batch_size))):
            # generate batch with random bit strings
            input_bit_string_batch = torch.tensor(
                np.random.choice([-1, 1], size=(len(batch), n))
            ).to(device)

            # calculate outputs
            synt_outputs = synt_net(input_bit_string_batch)
            gan_outputs = gen_net(synt_outputs)
            gan_outputs_with_background = batch.clone()
            gan_outputs_with_background[:, :, (m // 2):(m + m // 2), \
            (m // 2):(m + m // 2)] = gan_outputs
            rend_outputs = rend_net(gan_outputs_with_background)
            # rend_outputs = rend_net(synt_outputs)
            rec_outputs = rec_net(rend_outputs.to(device))

            # calculate accuracy
            errors = torch.cat([
                errors, torch.sum(
                    input_bit_string_batch == torch.sign(rec_outputs)
                    , axis=1)
            ])
        # print(errors)

    print('Mean accuracy: {}\n'.format(torch.mean(errors / n)))
