import torch
import sys

input_ckpt = sys.argv[1]
output_ckpt = sys.argv[2]

ckpt = torch.load(input_ckpt, map_location='cpu')['model_state_dict']
new_ckpt = {}
# for k, v in ckpt.items():
#     print('key old', k, v.shape)

for k, v in ckpt.items():
    # print(k)
    if k.startswith('edge_proj.'):
        k = k.replace('edge_proj.', 'edge_proj.fc.')
    elif k.startswith('node_proj.'):
        k = k.replace('node_proj.', 'node_proj.fc.')
    # elif k.startswith('edge_out.0'):
        # print(k)
        # k = k.replace('edge_out.', 'edge_out.fc.')
        # k = k.replace('edge_out.0', 'edge_mlp.fc.4')
    elif k.startswith('edge_out.'):
        # print(k)
        k = k.replace('edge_out.', 'edge_out.fc.')
        # k = k.replace('edge_out.2', 'edge_mlp.fc.6')
    elif k.startswith('gnn.convs.') and k.find('aggr_module') >= 0:
        print(k)
        continue
    elif k.startswith('edge_mlp.'):
        k = k.replace('edge_mlp.', 'edge_mlp.fc.')
        # print(k)
    new_ckpt[k] = v
# save new checkpoint
# for k, v in new_ckpt.items():
#     print('key new', k)
torch.save({'state_dict': new_ckpt, 'model_args': {'towers': 4, 'final_edge_update': True} }, output_ckpt)
print('Done!')
