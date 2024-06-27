import sys
import os


# DATASET = 'FT'
DATASET = 'DBLP'

'''PALE'''
# for i in range(5):
    # os.system(f'python PALE/runPALE.py --dataset {DATASET}  --adapt False --mapping_epochs 100')
    # os.system(f'python PALE/runPALE.py --dataset {DATASET}  --adapt True --support similarity --fast_lr 0.1 --mapping_epochs 100')


'''DeepLink'''
# for i in range(5):
#     # os.system(f'python DeepLink/run_deeplink.py --dataset {DATASET} --adapt False --supervised_epochs 40')
#     os.system(f'python DeepLink/run_deeplink.py --dataset {DATASET} --adapt True --support similarity --fast_lr 0.05 --supervised_epochs 15')
# os.system(f'python DeepLink/run_deeplink.py --dataset DBLP --adapt True --support similarity --fast_lr 0.05 --supervised_epochs 15')

'''Meta-NA'''
# for DATASET in ['FT', 'DBLP']:
#     for i in range(5):
#         os.system(f'python metaNA/runmetaNA.py --dataset {DATASET} --adapt False --epochs 40')
#     for i in range(5):
#         os.system(f'python metaNA/runmetaNA.py --dataset {DATASET} --adapt True --fast_lr 0.001 --epochs 40')

'''DegUIL'''
# for DATASET in ['FT', 'DBLP']:
# for i in range(5):
# os.system('python DegUIL/DegUIL_MANA.py --dataset FT --adapt True --fast_lr 0.1 --meta_bsz 4 --mu 0.001 --epochs 30 --meta_test_cluster 1 --device cuda:1')
# os.system('python DegUIL/DegUIL_MANA.py --dataset DBLP --adapt True --fast_lr 0.2 --meta_bsz 4 --mu 0.01 --epochs 10 --meta_test_cluster 1 --device cuda:3')
# for i in range(5):
    # os.system('python DegUIL/DegUIL_MANA.py --dataset FT --adapt False --mu 0.001 --meta_bsz 4 --epochs 30')
    # os.system('python DegUIL/DegUIL_MANA.py --dataset DBLP --adapt False --mu 0.01 --meta_bsz 4 --epochs 10 --device cuda:1')

# # support set
# for i in range(5):
#     # for support in ['random', 'neighbor']:
#     # os.system(
#     #     f'python DegUIL/DegUIL_MANA.py --dataset FT --support random --adapt True --fast_lr 0.1 --meta_bsz 4 --mu 0.001 --epochs 30')
#         os.system(
#             f'python DegUIL/DegUIL_MANA.py --dataset FT --support neighbor --adapt True --fast_lr 0.1 --meta_bsz 4 --mu 0.001 --epochs 30 --device cuda:0')


'''node2vec'''
# os.system(f'python Node2Vec/run_node2vec.py --dataset FT  --adapt False --mapping_epochs 70')
# os.system(f'python Node2Vec/run_node2vec.py --dataset DBLP  --adapt False --mapping_epochs 70')
# os.system(f'python Node2Vec/run_node2vec.py --dataset FT  --adapt True --fast_lr 0.2 --mapping_epochs 70')
# os.system(f'python Node2Vec/run_node2vec.py --dataset DBLP  --adapt True --fast_lr 0.5 --mapping_epochs 70')
# os.system(f'python Node2Vec/run_node2vec.py --dataset DBLP  --adapt True --fast_lr 0.05 --mapping_epochs 70')

# os.system(f'python Node2Vec/run_node2vec.py --dataset FT  --adapt True --fast_lr 0.05 --mapping_epochs 70')
# os.system(f'python Node2Vec/run_node2vec.py --dataset FT  --adapt True --fast_lr 0.2 --mapping_epochs 70')
# os.system(f'python Node2Vec/run_node2vec.py --dataset DBLP  --adapt True --fast_lr 0.05 --mapping_epochs 70')

# Strategy
# for i in range(5):
# os.system(f'python PALE/runPALE.py --dataset FT  --adapt True --support random --fast_lr 0.1 --mapping_epochs 100')
# os.system(f'python PALE/runPALE.py --dataset FT  --adapt True --support neighbor --fast_lr 0.1 --mapping_epochs 100')
# os.system(f'python DeepLink/run_deeplink.py --dataset FT --adapt True --support random --fast_lr 0.05 --supervised_epochs 20')
# os.system(f'python DeepLink/run_deeplink.py --dataset FT --adapt True --support neighbor --fast_lr 0.05 --supervised_epochs 20')

'''Size of support set (k-shot)'''
# # os.system(f'python DeepLink/run_deeplink.py --dataset DBLP --adapt True --fast_lr 0.01 --k_shot 1 --supervised_epochs 40')
# os.system(f'python DeepLink/run_deeplink.py --dataset DBLP --adapt True --fast_lr 0.01 --k_shot 2 --supervised_epochs 40')
# os.system(f'python DeepLink/run_deeplink.py --dataset DBLP --adapt True --fast_lr 0.01 --k_shot 5 --supervised_epochs 40')
# os.system(f'python DeepLink/run_deeplink.py --dataset DBLP --adapt True --fast_lr 0.01 --k_shot 10 --supervised_epochs 40')


'''JMAC'''
# os.chdir('./JMAC_EA')
#
# # DATASET = 'FT'
# # for i in range(5):
# #     os.system(f'python main_jmac_na.py --dataset {DATASET} --max_epoch 40 --alignment_learning_rate 0.0005')
# DATASET = 'DBLP'
# for i in range(5):
#     os.system(f'python main_jmac_na.py --dataset {DATASET} --max_epoch 15 --alignment_learning_rate 0.0003 --device cuda:1')

'''cluster meta-testing'''
# os.system(f'python DegUIL/DegUIL_MANA.py --dataset FT --adapt True --fast_lr 0.1 --meta_test_cluster 10 --meta_bsz 4 --mu 0.001 --epochs 30 --device cuda:1')
# for n_per_cluster in [1, 5, 10, 20, 50]:
#     os.system(f'python Node2Vec/run_node2vec.py --dataset FT --adapt True --meta_test_cluster {n_per_cluster} --support similarity --fast_lr 0.1 --mapping_epochs 70')
#     os.system(f'python Node2Vec/run_node2vec.py --dataset D_W_15K_V1 --adapt True --meta_test_cluster {n_per_cluster} --support similarity --meta_bsz 8 --mapping_lr 0.005 --fast_lr 0.1 --mapping_epochs 10')
# os.system(f'python Node2Vec/run_node2vec.py --dataset FT --adapt False --meta_test_cluster 1 --support similarity --mapping_lr 0.0005 --mapping_epochs 70')
# os.system(f'python Node2Vec/run_node2vec.py --dataset D_W_15K_V1 --adapt False --meta_test_cluster 1 --support similarity --meta_bsz 8 --mapping_lr 0.005 --mapping_epochs 30')
# for n_per_cluster in [1, 5, 10, 20, 50]:
#     os.system(f'python DeepLink/run_deeplink.py --dataset FT --adapt True --meta_test_cluster {n_per_cluster} --support similarity --fast_lr 0.05 --supervised_epochs 12 '
#               f'--batch_size_mapping 32 --unsupervised_epochs 20')
# os.system(f'python DeepLink/run_deeplink.py --dataset FT --adapt False --meta_test_cluster 1 --support similarity --supervised_epochs 40 '
#               f'--batch_size_mapping 32 --unsupervised_epochs 20')

# for n_per_cluster in [1, 5, 10, 20, 50]:
#     os.system(f'python DeepLink/run_deeplink.py --dataset D_W_15K_V1 --adapt True --meta_test_cluster {n_per_cluster} --support similarity --fast_lr 0.05 --supervised_epochs 15 '
#               f'--batch_size_mapping 500 --unsupervised_epochs 50 --device cuda:2')

# os.system(f'python DeepLink/run_deeplink.py --dataset D_W_15K_V1 --adapt False --meta_test_cluster 1 --support similarity --fast_lr 0.05 --supervised_epochs 40 '
#               f'--batch_size_mapping 500 --unsupervised_epochs 50 --device cuda:2')


'''robustness to noise'''
# for ptb_p in [5, 10, 15, 20, 25]:
    # ptb_p = ptb_p/100
    # os.system(f'python Node2Vec/run_n2v_noise.py --dataset FT --adapt False --ptb_p {ptb_p} --mapping_lr 0.0005 --mapping_epochs 70')
    # os.system(f'python Node2Vec/run_n2v_noise.py --dataset FT --adapt True --ptb_p {ptb_p} --meta_test_cluster 1 --support similarity --mapping_lr 0.0005 --mapping_epochs 70 --device cuda:2')
