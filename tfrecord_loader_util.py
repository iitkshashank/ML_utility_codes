import tensorflow as tf
import numpy as np

class tfrecord_data_util():
	def __init__(self,filenames,buffer_size=None,num_parallel_reads=None):
		self.filenames=filenames
		self.buffer_size=None
		self.num_parallel_reads=None
		self.length=None
		self.raw_dataset=tf.data.TFRecordDataset(filenames,buffer_size,num_parallel_reads)
		self.dataset=tf.data.TFRecordDataset(filenames,buffer_size,num_parallel_reads)
	def __len__(self):
		if self.length==None:
			self.length=sum(1 for i in self.raw_dataset)
		return self.length
	def head(self,num):
		for i in self.dataset:
			return i
	def prediction(self,model):
		preds=[]
		target=[]
		for batch in self.dataset:
			x=batch[0]
			y=batch[1]
			if len(preds)==0:
				preds.append(model.predict(x))
				target.append(y)
			else:
				preds[0]=np.append(preds[0],model.predict(x),axis=0)
				target[0]=np.append(target[0],y,axis=0)
		self.target=target[0]
		self.pred=preds[0]
		return preds[0],target[0]
	def evaluation(self,performance_fn,model=None):
		if model==None:
			return performance_fn(self.target,self,self.preds)
		_,_=self.prediction(model)
		self.evaluation(performance_fn)


