import io
import os
import torch
#from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from ml_things import plot_dict, plot_confusion_matrix, fix_text
from sklearn.metrics import classification_report, accuracy_score
from transformers import (set_seed, TrainingArguments, Trainer, GPT2Config, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup, GPT2ForSequenceClassification)


set_seed(123)
epochs = 3
batch_size = 1
max_length = None
# Look for gpu to use. Will use `cpu` by default if no gpu found.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#os.environ['CUDA_VISIBLE_DEVICES']='0, 1, 2, 3'
#device = 'cpu'
model_name_or_path = 'gpt2'
# Dictionary of labels and their id - this will be used to convert string labels to number ids.
#labels_ids = {'neg': 0, 'pos': 1}
labels_ids = {'0': 0, '1': 1}
n_labels = len(labels_ids)


class ReadKG(Dataset):
	def __init__(self, path, use_tokenizer):
		self.texts = []
		self.labels = []
		text_path = os.path.join(path, 'description.txt')
		with open(text_path, encoding='utf-8') as f:
			lines = f.readlines()
			for line in lines:
				line = line.strip()
				line = line.replace('\t', '')
				line = line.replace('  ', ' ')
				content = fix_text(line)
				self.texts.append(content)

		labels_path = os.path.join(path, 'labels.txt')
		with open(labels_path, encoding='utf-8') as f:
			lines = f.readlines()
			for line in lines:
				line = line.strip()
				line = line.replace('\t', '')
				line = line.replace('  ', ' ')
				self.labels.append(line)
			
# Number of examples.
		self.n_examples = len(self.labels)
		return
	
	def __len__(self):
		return self.n_examples

	def __getitem__(self, item):
		return {'text':self.texts[item],'label':self.labels[item]}
		


class Gpt2ClassificationCollator(object):
	def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None):

# Tokenizer to be used inside the class.
		self.use_tokenizer = use_tokenizer
# Check max sequence length.
		self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
	# Label encoder used inside the class.
		self.labels_encoder = labels_encoder
		return

	def __call__(self, sequences):
		# Get all texts from sequences list.
		texts = [sequence['text'] for sequence in sequences]
		# Get all labels from sequences list.
		labels = [sequence['label'] for sequence in sequences]
		# Encode all labels using label encoder.
		labels = [self.labels_encoder[label] for label in labels]
		#Call tokenizer on all texts to convert into tensors of numbers with appropriate padding.
		inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)
		# Update the inputs with the associated encoded labels as tensor.
		inputs.update({'labels':torch.tensor(labels)})

		return inputs



def train(dataloader, optimizer_, scheduler_, device_):

  # Use global variable for model.
	global model

  # Tracking variables.
	predictions_labels = []
	true_labels = []
  # Total loss for this epoch.
	total_loss = 0

  # Put the model into training mode.
	model.train()

  # For each batch of training data...
	for batch in dataloader:

	# Add original labels - use later for evaluation.
		true_labels += batch['labels'].numpy().flatten().tolist()
 
	# move batch to device
		batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}

	# Always clear any previously calculated gradients before performing a
	# backward pass.
		model.zero_grad()

	# Perform a forward pass (evaluate the model on this training batch).
	# This will return the loss (rather than the model output) because we
	# have provided the `labels`.
	# The documentation for this a bert model function is here: 
 	# https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
		outputs = model(**batch)

	# The call to `model` always returns a tuple, so we need to pull the 
# loss value out of the tuple along with the logits. We will use logits
# later to calculate training accuracy.
		loss, logits = outputs[:2]

# Accumulate the training loss over all of the batches so that we can
# calculate the average loss at the end. `loss` is a Tensor containing a
# single value; the `.item()` function just returns the Python value 
# from the tensor.
		total_loss += loss.item()

# Perform a backward pass to calculate the gradients.
		loss.backward()

# Clip the norm of the gradients to 1.0.
# This is to help prevent the "exploding gradients" problem.
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# Update parameters and take a step using the computed gradient.
# The optimizer dictates the "update rule"--how the parameters are
# modified based on their gradients, the learning rate, etc.
		optimizer.step()

# Update the learning rate.
		scheduler.step()
	
# Move logits and labels to CPU
		logits = logits.detach().cpu().numpy()

# Convert these logits to list of predicted labels values.
		predictions_labels += logits.argmax(axis=-1).flatten().tolist()

  # Calculate the average loss over the training data.
	avg_epoch_loss = total_loss / len(dataloader)
  
  # Return all true labels and prediction for future evaluations.
	return true_labels, predictions_labels, avg_epoch_loss



def validation(dataloader, device_):
  # Use global variable for model.
	global model

  # Tracking variables
	predictions_labels = []
	true_labels = []
  #total loss for this epoch.
	total_loss = 0

  # Put the model in evaluation mode--the dropout layers behave differently
  # during evaluation.
	model.eval()

  # Evaluate data for one epoch
	for batch in dataloader:

	# add original labels
		true_labels += batch['labels'].numpy().flatten().tolist()

# move batch to device
		batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}
	
# Telling the model not to compute or store gradients, saving memory and
# speeding up validation
		with torch.no_grad():

# Forward pass, calculate logit predictions.
# This will return the logits rather than the loss because we have
# not provided labels.
# token_type_ids is the same as the "segment ids", which 
# differentiates sentence 1 and 2 in 2-sentence tasks.
# The documentation for this `model` function is here: 

			outputs = model(**batch)

# The call to `model` always returns a tuple, so we need to pull the 
# loss value out of the tuple along with the logits. We will use logits
# later to to calculate training accuracy.
			loss, logits = outputs[:2]

# Move logits and labels to CPU
			logits = logits.detach().cpu().numpy()

# Accumulate the training loss over all of the batches so that we can
# calculate the average loss at the end. `loss` is a Tensor containing a
# single value; the `.item()` function just returns the Python value 
# from the tensor.
			total_loss += loss.item()

# get predicitons to list
			predict_content = logits.argmax(axis=-1).flatten().tolist()

# update list
			predictions_labels += predict_content
		
  # Calculate the average loss over the training data.
		avg_epoch_loss = total_loss / len(dataloader)

  # Return all true labels and prediciton for future evaluations.
	return true_labels, predictions_labels, avg_epoch_loss


"""## **Load Model and Tokenizer**"""
# Get model configuration.
print('Loading configuraiton...')
model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name_or_path, num_labels=n_labels)

# Get model's tokenizer.
print('Loading tokenizer...')
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
# default to left padding
tokenizer.padding_side = "left"
# Define PAD Token = EOS Token = 50256
tokenizer.pad_token = tokenizer.eos_token

# Get the actual model.
print('Loading model...')
model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path, config=model_config)

# resize model embedding to match new tokenizer
model.resize_token_embeddings(len(tokenizer))

# fix model padding token id
model.config.pad_token_id = model.config.eos_token_id

# Load model to defined device.
model.to(device)
print('Model loaded to `%s`'%device)




"""## **Dataset and Collator**"""
# Create data collator to encode text and labels into numbers.
gpt2_classificaiton_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer, 
                                                          labels_encoder=labels_ids, 
                                                          max_sequence_len=max_length)
print('Dealing with Train...')
# Create pytorch dataset.
train_dataset = ReadKG(path='./data_for_GPT2/FB13/train', 
                               use_tokenizer=tokenizer)
print('Created `train_dataset` with %d examples!'%len(train_dataset))

# Move pytorch dataset into dataloader.
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=gpt2_classificaiton_collator)
print('Created `train_dataloader` with %d batches!'%len(train_dataloader))

print()

print('Dealing with Validation...')
# Create pytorch dataset.
valid_dataset =  ReadKG(path='./data_for_GPT2/FB13/dev', 
                               use_tokenizer=tokenizer)
print('Created `valid_dataset` with %d examples!'%len(valid_dataset))

# Move pytorch dataset into dataloader.
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=gpt2_classificaiton_collator)
print('Created `eval_dataloader` with %d batches!'%len(valid_dataloader))


print('Dealing with Test...')
# Create pytorch dataset.
test_dataset =  ReadKG(path='./data_for_GPT2/FB13/test', 
                               use_tokenizer=tokenizer)
print('Created `test_dataset` with %d examples!'%len(test_dataset))

# Move pytorch dataset into dataloader.
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=gpt2_classificaiton_collator)
print('Created `eval_dataloader` with %d batches!'%len(test_dataloader))





"""## **Train**"""

# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(),
                  lr = 5e-5, # default is 5e-5
                  eps = 1e-8 # default is 1e-8.
                  )

# Total number of training steps is number of batches * number of epochs.
# `train_dataloader` contains batched data so `len(train_dataloader)` gives 
# us the number of batches.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

# Store the average loss after each epoch so we can plot them.
all_loss = {'train_loss':[], 'val_loss':[]}
all_acc = {'train_acc':[], 'val_acc':[]}

# Loop through each epoch.
print('Epoch')
for epoch in range(epochs):
  print()
  print('Training on batches...')
  # Perform one full pass over the training set.
  train_labels, train_predict, train_loss = train(train_dataloader, optimizer, scheduler, device)
  train_acc = accuracy_score(train_labels, train_predict)

  # Get prediction form model on validation data. 
  print('Validation on batches...')
  valid_labels, valid_predict, val_loss = validation(valid_dataloader, device)
  val_acc = accuracy_score(valid_labels, valid_predict)

  # Print loss and accuracy values to see how training evolves.
  print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f"%(train_loss, val_loss, train_acc, val_acc))
  print()

  # Store the loss value for plotting the learning curve.
  all_loss['train_loss'].append(train_loss)
  all_loss['val_loss'].append(val_loss)
  all_acc['train_acc'].append(train_acc)
  all_acc['val_acc'].append(val_acc)

# Plot loss curves.
plot_dict(all_loss, use_xlabel='Epochs', use_ylabel='Value', use_linestyles=['-', '--'])

# Plot accuracy curves.
plot_dict(all_acc, use_xlabel='Epochs', use_ylabel='Value', use_linestyles=['-', '--'])



"""## **Evaluate**"""

# Get prediction form model on validation data. This is where you should use your test data.
true_labels, predictions_labels, avg_epoch_loss = validation(test_dataloader, device)
test_acc = accuracy_score(true_labels, predictions_labels)

print("  test_loss: %.5f - test_acc: %.5f "%(avg_epoch_loss, test_acc))
print()
print(true_labels,predictions_labels)
# Create the evaluation report.
evaluation_report = classification_report(true_labels, predictions_labels, labels=list(labels_ids.values()), target_names=list(labels_ids.keys()))
# Show the evaluation report.
print(evaluation_report)

# Plot confusion matrix.
plot_confusion_matrix(y_true=true_labels, y_pred=predictions_labels, 
                      classes=list(labels_ids.keys()), normalize=True, 
                      magnify=0.1,
                      );

