import numpy as np
from sklearn.manifold import TSNE
from argparse import ArgumentParser


parser = ArgumentParser(description='tsne visulization')
parser.add_argument('--model',    type=str,   default='refocs', help='model peeler or refocs')
args = parser.parse_args()
k=5
if args.model == 'refocs':
	path = '/home/snag005/Desktop/fs_ood/trial2/models/gtsrb2TT100K/tsne/refocs/'
	support_emb = np.load(path+'support_emb.npy',mmap_mode=None, allow_pickle=True)
	exemplar_emb = np.load(path+'exemplar_emb.npy',mmap_mode=None, allow_pickle=True)
	in_query_emb = np.load(path+'in_query_emb.npy',mmap_mode=None, allow_pickle=True)	
	mod_in_query_emb = np.load(path+'mod_in_query_emb.npy',mmap_mode=None, allow_pickle=True)	
	out_query_emb = np.load(path+'out_query_emb.npy',mmap_mode=None, allow_pickle=True)	
	mod_out_query_emb = np.load(path+'mod_out_query_emb.npy',mmap_mode=None, allow_pickle=True)
	total_emb = np.concatenate((support_emb,exemplar_emb,in_query_emb,out_query_emb),axis=0)
	total_emb_mod = np.concatenate((support_emb,exemplar_emb,mod_in_query_emb,mod_out_query_emb),axis=0)
	tsne_refocs = TSNE(n_components=2).fit_transform(total_emb)	
	tsne_refocs_mod = TSNE(n_components=2).fit_transform(total_emb_mod)
	# tsne_support_emb_refocs = []
	# tsne_in_query_emb_refocs  = []
	# tsne_out_query_emb_refocs = []
	# tsne_mod_in_query_emb_refocs = []
	# tsne_mod_out_query_emb_refocs = []
	# tsne_exemplar_emb_refocs = []
	# for i in range(0,k):
	# 	ee = exemplar_emb[i,:]
	# 	se = support_emb[20*i:(20+20*i),:]
	# 	iqe = in_query_emb[20*i:(20+20*i),:]
	# 	oqe = out_query_emb[20*i:(20+20*i),:]
	# 	mique = mod_in_query_emb[20*i:(20+20*i),:]
	# 	moque = mod_out_query_emb[20*i:(20+20*i),:]
	# 	tsne_support_emb_refocs.append(TSNE(n_components=2).fit_transform(se))
	# 	tsne_in_query_emb_refocs.append(TSNE(n_components=2).fit_transform(iqe))
	# 	tsne_out_query_emb_refocs.append(TSNE(n_components=2).fit_transform(oqe))
	# tsne_support_emb_peeler = np.asarray(tsne_support_emb_peeler)
	# tsne_in_query_emb_peeler = np.asarray(tsne_in_query_emb_peeler)
	# tsne_out_query_emb_peeler = np.asarray(tsne_out_query_emb_peeler)
	# np.save(path+'tsne_support_emb_peeler.npy', tsne_support_emb_peeler, allow_pickle=True, fix_imports=True)
	# np.save(path+'tsne_in_query_emb_peeler.npy', tsne_in_query_emb_peeler, allow_pickle=True, fix_imports=True)
	np.save(path+'tsne_refocs.npy', tsne_refocs, allow_pickle=True, fix_imports=True)
	np.save(path+'tsne_refocs_mod.npy',tsne_refocs_mod,allow_pickle=True,fix_imports=True)

elif args.model == 'peeler':
	path = '/home/snag005/Desktop/fs_ood/trial2/models/gtsrb2TT100K/tsne/peeler/'
	support_emb = np.load(path+'support_emb.npy',mmap_mode=None, allow_pickle=True)
	support_emb = np.reshape(support_emb,(support_emb.shape[1],support_emb.shape[2]))
	query_emb = np.load(path+'query_emb.npy',mmap_mode=None, allow_pickle=True)	
	query_emb = np.reshape(query_emb,(query_emb.shape[1],query_emb.shape[2]))
	total_emb = np.concatenate((support_emb,query_emb))
	tsne_peeler = TSNE(n_components=2).fit_transform(total_emb)
	# in_query_emb = query_emb[:100,:]
	# out_query_emb = query_emb[100:,:]
	# tsne_support_emb_peeler = []
	# tsne_in_query_emb_peeler = []
	# tsne_out_query_emb_peeler = []
	# for i in range(0,k):
	# 	se = support_emb[20*i:(20+20*i),:]
	# 	iqe = in_query_emb[20*i:(20+20*i),:]
	# 	oqe = out_query_emb[20*i:(20+20*i),:]
	# 	tsne_support_emb_peeler.append(TSNE(n_components=2).fit_transform(se))
	# 	tsne_in_query_emb_peeler.append(TSNE(n_components=2).fit_transform(iqe))
	# 	tsne_out_query_emb_peeler.append(TSNE(n_components=2).fit_transform(oqe))
	# tsne_support_emb_peeler = np.asarray(tsne_support_emb_peeler)
	# tsne_in_query_emb_peeler = np.asarray(tsne_in_query_emb_peeler)
	# tsne_out_query_emb_peeler = np.asarray(tsne_out_query_emb_peeler)
	# np.save(path+'tsne_support_emb_peeler.npy', tsne_support_emb_peeler, allow_pickle=True, fix_imports=True)
	# np.save(path+'tsne_in_query_emb_peeler.npy', tsne_in_query_emb_peeler, allow_pickle=True, fix_imports=True)
	np.save(path+'tsne_peeler.npy', tsne_peeler, allow_pickle=True, fix_imports=True)

