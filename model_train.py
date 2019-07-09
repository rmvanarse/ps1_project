

from lib import *
from model import SiameseModel
from dataset import SignatureDataset

# import model_preprocess

train_master_index = []
test_master_index = []

iter_list = []                           # Saves Iterations at which the model has been evaluated
train_loss_list = []                     # Saves Train Loss
train_acc_list = []                      # Saves Train Accuracy
test_loss_list = []                      # Saves Test Loss
test_acc_list = []                       # Saves Test Accuracy

def_triplet_loss_margin = 1
def_learning_rate = 0.001
def_batch_size = 100
def_n_iters = 760
def_inspect_size = 15


############################### FUNCTIONS ###############################

def get_encodings(matrix):

	'''
	Accepts a Tensor and returns its encoding.
	'''
	
	if torch.cuda.is_available():
		matrix = Variable(matrix.cuda())    
	else:
		matrix = Variable(matrix)
	
	matrix = matrix.float()
	matrix_enc = model(matrix)
	
	return matrix_enc



def return_diff(anchors_enc, positives_enc, negatives_enc):
 
	'''
	Accepts the encodings of three tensors.
	
	Returns d(E1,E2) and d(E1,E3) where d(A,B) is the 
	Frobenius norm of the vector A-B.
	
	Returns the result as a pair of numpy arrays.
	'''

	assert(anchors_enc.shape == positives_enc.shape)
	assert(anchors_enc.shape == negatives_enc.shape)
	
	num = anchors_enc.shape[0]
	pos_diff_vec = []
	neg_diff_vec = []
	
	anchors_enc = anchors_enc.cpu().detach().numpy()
	positives_enc = positives_enc.cpu().detach().numpy()
	negatives_enc = negatives_enc.cpu().detach().numpy()
	
	for i in range(num):
		pos_diff = np.linalg.norm(anchors_enc[i] - positives_enc[i])
		neg_diff = np.linalg.norm(anchors_enc[i] - negatives_enc[i])
		
		pos_diff_vec.append(pos_diff)
		neg_diff_vec.append(neg_diff)
   
	return np.array(pos_diff_vec), np.array(neg_diff_vec)    


def get_time(time_begin, time_end):

	'''
	Formats the time elapsed during training.
	Returns hours, minutes, and seconds.
	'''
		
	FMT = '%H:%M:%S'
	td = (datetime.strptime(time_end[11:19], FMT) - datetime.strptime(time_begin[11:19], FMT)).seconds
	hr = (td//3600)
	min = (td - 3600*hr)//60
	sec = (td - 3600*hr - 60*min)

	return hr, min, sec



############################### MAIN ###############################

if __name__ == "__main__":

	# Parsing arguments.

	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--n-iters", 
		"-n", 
		default=def_n_iters,
		type=int,
		help="Number of iteration to train the model."
	)
	
	parser.add_argument(
		"--inspect-size", 
		"-i", 
		default=def_inspect_size,
		type=int,
		help="Time period after which the model is evaluated against the test set."
	)

	parser.add_argument(
		"--learning-rate", 
		"-r",
		default=def_learning_rate,
		type=float, 
		help="Simply the learning rate."
	)

	parser.add_argument(
		"--batch-size", 
		"-b", 
		default=def_batch_size,
		type=int,
		help="Simply the batch size."
	)

	parser.add_argument(
		"--triplet-loss-margin", 
		"-t", 
		default=def_triplet_loss_margin,
		type=int,
		help="Margin for the triplet loss."
	)

	args = parser.parse_args()

	# __________________________________________________________________________________

	# Loading index lists
	fileObject = open('master_indices.pkl','rb')
	listf = pickle.load(fileObject) 
	train_master_index = listf[0]
	test_master_index = listf[1]

	# Instantiating the dataset class
	train_dataset = SignatureDataset(train_master_index, test_master_index, is_train = True)
	test_dataset = SignatureDataset(train_master_index, test_master_index, is_train = False)

	# Making dataset iterable
	train_loader = torch.utils.data.DataLoader(
		dataset=train_dataset, 
		batch_size=args.batch_size, 
		# shuffle=True
	)

	test_loader = torch.utils.data.DataLoader(
		dataset=test_dataset, 
		batch_size=args.batch_size, 
		# shuffle=True
	)

	# Instantiating the model class
	model = SiameseModel()

	if torch.cuda.is_available():
		model.cuda()

	# Instantiating the loss and optimizer class
	triplet_loss = nn.TripletMarginLoss(margin=args.triplet_loss_margin)
	optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)


	# __________________________________________________________________________________

	# Initializations

	iterr = 0
	iter_list.clear()                      
	train_loss_list.clear()                  
	train_acc_list.clear()                  
	test_loss_list.clear()                  
	test_acc_list.clear()          

	num_epochs = int(args.n_iters / (len(train_dataset) / args.batch_size))

	print("Number of Iterations     :", args.n_iters)
	print("Number of Epochs         :", num_epochs)
	print("Number of Sample-Points  :", int(args.n_iters/args.inspect_size))
	print("------------------------------------------------")

	# __________________________________________________________________________________

	### TRAINING THE MODEL ###

	time_begin = time.asctime()   # Time when training started

	init_iters = iterr

	for epoch in range(num_epochs):
		for i, (anchors, positives, negatives) in enumerate(train_loader):
			
			# Get encodings by forward propogation
			anchors_enc = get_encodings(anchors)
			positives_enc = get_encodings(positives) 
			negatives_enc = get_encodings(negatives)
			   
			# Clearing the previous gradients
			optimizer.zero_grad()                                        

			 # Calculating the Train loss
			loss = triplet_loss(anchors_enc, positives_enc, negatives_enc)
			
			# Backward propogation
			loss.backward()
		   
			# Optimizing the parameters
			optimizer.step()                                            
			
			iterr += 1
			
			print("Iter {:.0f} Done.\t Loss : {:.5f}".format(iterr - init_iters, loss.item()))
			
			
			# -----------------------------------------------------------------------------------
			### Inspecting the performance of the model ###
			
			if (iterr == 0 or iterr % args.inspect_size == 0):
			
				iter_list.append(iterr)
				print("Iteration : {:.0f}/{:.0f} [{:2.0f}%] ".format(iterr - init_iters, args.n_iters, 100*(iterr - init_iters)/args.n_iters))
				print('---------------------------')
				
			# -----------------------------------------------------------------------------------
			### Calculating train accuracy and loss ###                                          

				# NOTE : Using encoding obtained in current training iteration.
			
				# Append train loss
				train_loss_list.append(loss.item())
				
				# Use encoding to obtain vector difference
				pos_diff, neg_diff = return_diff(anchors_enc, positives_enc, negatives_enc)
				
				# Append train accuracy
				num_sample = anchors.shape[0]
				tot_correct = np.sum(pos_diff < neg_diff)
				train_acc = tot_correct/num_sample * 100
				train_acc_list.append(train_acc)
				
				print('[Train]\t Loss: {:.5f} | Acc: {:2.0f}%'.format(loss.item(), train_acc)) 
			
			# -----------------------------------------------------------------------------------
			### Calculating test accuracy and loss ###
				
				# Use 100 samples for inspection from test set
				anchors, positives, negatives = next(iter(test_loader))

				# Get encodings by forward propogation
				anchors_enc = get_encodings(anchors)
				positives_enc = get_encodings(positives) 
				negatives_enc = get_encodings(negatives)

				# Append test loss
				loss = triplet_loss(anchors_enc, positives_enc, negatives_enc)        
				test_loss_list.append(loss.item())                  
					  
				# Use encoding to obtain vector difference
				pos_diff, neg_diff = return_diff(anchors_enc, positives_enc, negatives_enc)
				
				# Append test accuracy 
				num_sample = anchors.shape[0]
				tot_correct = np.sum(pos_diff < neg_diff)
				test_acc = tot_correct/num_sample * 100
				test_acc_list.append(test_acc)
				
				print('[Test ]\t Loss: {:.5f} | Acc: {:2.0f}%'.format(loss.item(), test_acc))        
			
			# -----------------------------------------------------------------------------------
				print('=========================================================')
				
				
	print("\nTraining Done.")

	# Time when training ended
	time_end = time.asctime()


	# __________________________________________________________________________________

	# Training output.

	hr, min, sec = get_time(time_begin, time_end)

	print("Total Iterations     : {:.0f}".format(iterr))
	print("Total Epochs         : {:.0f}".format(iterr*100/60000))
	print("Total Sample-Points  : {:.0f}".format(iterr/args.inspect_size))
	print("-------------------------------")
	print("Loss - Train     : {:.2f}".format(np.mean(train_loss_list[-10:])))
	print("Loss - Test      : {:.2f}".format(np.mean(test_loss_list[-10:])))
	print("Acc - Train     : {:.2f}".format(np.mean(train_acc_list[-10:])))
	print("Acc - Test      : {:.2f}".format(np.mean(test_acc_list[-10:])))
	print("-------------------------------")
	print("Start Time        : {}".format(time_begin[11:19]))
	print("End Time          : {}".format(time_end[11:19]))
	print("Total Train-time  : {:2.0f}:{:2.0f}:{:2.0f}".format(hr,min,sec))