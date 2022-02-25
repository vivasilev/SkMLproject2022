import torch
import numpy as np

from tqdm import tqdm


def test(device, num_strings, batch_size,
         synt_net, rec_net, n):
    
    errors = torch.tensor([])
    
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i in tqdm(range(int(num_strings / batch_size))):
            # generate batch with random bit strings
            input_bit_string_batch = torch.tensor(
                np.random.choice([-1, 1], size=(batch_size, n))
            ).to(device)
            
            # calculate outputs
            synt_outputs = synt_net(input_bit_string_batch)
            rec_outputs = rec_net(synt_outputs)
            
            #print('input: ', input_bit_string_batch[0])
            #print('rec: ', torch.sign(rec_outputs[0]))
            
            # calculate accuracy
            errors = torch.cat([
                errors, torch.sum(
                    input_bit_string_batch == torch.sign(rec_outputs)
                    , axis=1)
            ])
            #print(errors)
            
    print('Mean accuracy: {}\n'.format(torch.mean(errors / n)))

