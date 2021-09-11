import os
import numpy as np 
# import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE
from sklearn import decomposition


def plot_peeler():
	s = np.load('/home/snag005/Desktop/fs_ood/trial2/models/gtsrb2TT100K/tsne/peeler/support_emb.npy').squeeze()
	q = np.load('/home/snag005/Desktop/fs_ood/trial2/models/gtsrb2TT100K/tsne/peeler/query_emb.npy').squeeze()
	q_i = q[:250]
	q_o = q[250:]
	print(s.shape, q_i.shape, q_o.shape)
	y_s = np.hstack([i*np.ones(50,dtype=np.int8)*7 for i in range(5)])
	y_q_i = np.hstack([i*np.ones(50,dtype=np.int8)*7 for i in range(5)])
	y_q_o = np.hstack([50 for _ in range(250)])
	X = np.vstack([s,q_i,q_o])
	Y = np.hstack([y_s,y_q_i,y_q_o])
	print(Y)
	pca = decomposition.PCA(n_components=50)
	pca.fit(X)
	# classes = ['1','2','3','4','5','Open Classes']
	X = pca.transform(X)
	plt.figure()
	tsne = TSNE(n_components=2, verbose=0, perplexity=50, n_iter=2000)
	X_ = tsne.fit_transform(X)
	scatter = plt.scatter(X_[:,0],X_[:,1],c=Y,s=20,alpha=0.4)
	plt.axis('off')
	plt.savefig('peeler.pdf', format='pdf', dpi=1200, bbox_inches = 'tight', pad_inches = 0)
	plt.show()
	

def plot_refocs():
	s = np.load('/home/snag005/Desktop/fs_ood/trial2/models/gtsrb2TT100K/tsne/refocs_new/support_emb.npy').squeeze()
	q_i = np.load('/home/snag005/Desktop/fs_ood/trial2/models/gtsrb2TT100K/tsne/refocs_new/in_query_emb.npy').squeeze()
	q_o = np.load('/home/snag005/Desktop/fs_ood/trial2/models/gtsrb2TT100K/tsne/refocs_new/out_query_emb.npy').squeeze()
	print(s.shape, q_i.shape, q_o.shape)
	y_s = np.hstack([i*np.ones(50,dtype=np.int8)*7 for i in range(5)])
	y_q_i = np.hstack([i*np.ones(50,dtype=np.int8)*7 for i in range(5)])
	y_q_o = np.hstack([50 for _ in range(250)])
	X = np.vstack([s,q_i,q_o])
	Y = np.hstack([y_s,y_q_i,y_q_o])
	print(Y)
	pca = decomposition.PCA(n_components=50)
	pca.fit(X)
	# classes = ['1','2','3','4','5','Open Classes']
	X = pca.transform(X)
	plt.figure()
	tsne = TSNE(n_components=2, verbose=0, perplexity=50, n_iter=2000)
	X_ = tsne.fit_transform(X)
	scatter = plt.scatter(X_[:,0],X_[:,1],c=Y,s=20,alpha=0.4)
	# plt.legend(handles=scatter.legend_elements()[0], labels=classes)
	plt.axis('off')
	plt.savefig('refocs.pdf', format='pdf', dpi=1200, bbox_inches = 'tight', pad_inches = 0)
	plt.show()

def plot_proto():
	s = np.load('/home/snag005/Desktop/fs_ood/trial2/models/gtsrb2TT100K/tsne/protonet/support_emb.npy')
	q_i = np.load('/home/snag005/Desktop/fs_ood/trial2/models/gtsrb2TT100K/tsne/protonet/in_query_emb.npy').squeeze()
	q_o = np.load('/home/snag005/Desktop/fs_ood/trial2/models/gtsrb2TT100K/tsne/protonet/out_query_emb.npy').squeeze()
	print(s.shape, q_i.shape, q_o.shape)
	y_s = np.hstack([i*np.ones(50,dtype=np.int8)*7 for i in range(5)])
	y_q_i = np.hstack([i*np.ones(50,dtype=np.int8)*7 for i in range(5)])
	y_q_o = np.hstack([50 for _ in range(250)])
	X = np.vstack([s,q_i,q_o])
	Y = np.hstack([y_s,y_q_i,y_q_o])
	print(Y)
	pca = decomposition.PCA(n_components=50)
	pca.fit(X)
	# classes = ['1','2','3','4','5','Open Classes']
	X = pca.transform(X)
	plt.figure()
	tsne = TSNE(n_components=2, verbose=0, perplexity=30, n_iter=2000)
	X_ = tsne.fit_transform(X)
	scatter = plt.scatter(X_[:,0],X_[:,1],c=Y,s=20,alpha=0.4)
	# plt.legend(handles=scatter.legend_elements()[0], labels=classes)
	plt.axis('off')
	plt.savefig('proto.pdf', format='pdf', dpi=1200, bbox_inches = 'tight', pad_inches = 0)
	
	plt.show()

def plot_refocs_mod():
	s = np.load('/home/snag005/Desktop/fs_ood/trial2/models/gtsrb2TT100K/tsne/refocs_new/support_emb.npy').squeeze()
	q_i = np.load('/home/snag005/Desktop/fs_ood/trial2/models/gtsrb2TT100K/tsne/refocs_new/mod_in_query_emb.npy').squeeze()
	q_o = np.load('/home/snag005/Desktop/fs_ood/trial2/models/gtsrb2TT100K/tsne/refocs_new/mod_out_query_emb.npy').squeeze()
	print(s.shape, q_i.shape, q_o.shape)
	y_s = np.hstack([i*np.ones(50,dtype=np.int8)*7 for i in range(5)])
	y_q_i = np.hstack([i*np.ones(50,dtype=np.int8)*7 for i in range(5)])
	y_q_o = np.hstack([50 for _ in range(250)])
	X = np.vstack([s,q_i,q_o])
	Y = np.hstack([y_s,y_q_i,y_q_o])
	print(Y)
	pca = decomposition.PCA(n_components=50)
	pca.fit(X)
	# classes = ['1','2','3','4','5','Open Classes']
	X = pca.transform(X)
	plt.figure()
	tsne = TSNE(n_components=2, verbose=0, perplexity=30, n_iter=2000)
	X_ = tsne.fit_transform(X)
	scatter = plt.scatter(X_[:,0],X_[:,1],c=Y,s=20,alpha=0.4)
	# plt.legend(handles=scatter.legend_elements()[0], labels=classes)
	plt.axis('off')
	plt.savefig('refocs_mod.pdf', format='pdf', dpi=1200, bbox_inches = 'tight', pad_inches = 0)
	plt.axis('off')
	plt.show()



method = 'peeler'

if method == 'peeler':
	plot_peeler()
elif method == 'refocs':
	plot_refocs()
elif method == 'proto':
	plot_proto()
elif method == 'refocs_mod':
	plot_refocs_mod()
else:
	print('Error!')

