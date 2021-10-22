import io
import os
import torch
from torch.utils.data import Dataset, DataLoader
from ml_things import plot_dict, plot_confusion_matrix, fix_text
from sklearn.metrics import classification_report, accuracy_score
from transformers import (set_seed, TrainingArguments, Trainer, GPT2Config, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup, GPT2ForSequenceClassification)


set_seed(123)
epochs = 3
batch_size = 1
max_length = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#os.environ['CUDA_VISIBLE_DEVICES']='0, 1, 2, 3'
#device = 'cpu'
model_name_or_path = 'gpt2'
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
			
		self.n_examples = len(self.labels)
		return
	
	def __len__(self):
		return self.n_examples

	def __getitem__(self, item):
		return {'text':self.texts[item],'label':self.labels[item]}
		


class Gpt2ClassificationCollator(object):
	def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None):
		self.use_tokenizer = use_tokenizer
		self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
		self.labels_encoder = labels_encoder
		return

	def __call__(self, sequences):
		texts = [sequence['text'] for sequence in sequences]
		labels = [sequence['label'] for sequence in sequences]
		labels = [self.labels_encoder[label] for label in labels]
		inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)
		inputs.update({'labels':torch.tensor(labels)})
		return inputs



def train(dataloader, optimizer_, scheduler_, device_):
	global model
	predictions_labels = []
	true_labels = []
	total_loss = 0
	model.train()
	for batch in dataloader:
		true_labels += batch['labels'].numpy().flatten().tolist()
 		batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}
		model.zero_grad()
		outputs = model(**batch)
		loss, logits = outputs[:2]
		total_loss += loss.item()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		optimizer.step()
		scheduler.step()
		logits = logits.detach().cpu().numpy()
		predictions_labels += logits.argmax(axis=-1).flatten().tolist()
	avg_epoch_loss = total_loss / len(dataloader)
 	return true_labels, predictions_labels, avg_epoch_loss



def validation(dataloader, device_):
	global model
	predictions_labels = []
	true_labels = []
	total_loss = 0
	model.eval()
	for batch in dataloader:
		true_labels += batch['labels'].numpy().flatten().tolist()
		batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}
		with torch.no_grad():
			outputs = model(**batch)
			loss, logits = outputs[:2]
			logits = logits.detach().cpu().numpy()
			total_loss += loss.item()
			predict_content = logits.argmax(axis=-1).flatten().tolist()
			predictions_labels += predict_content
		avg_epoch_loss = total_loss / len(dataloader)
	return true_labels, predictions_labels, avg_epoch_loss

print('Loading configuraiton...')
model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name_or_path, num_labels=n_labels)

print('Loading tokenizer...')
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
print('Loading model...')
model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path, config=model_config)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id
odel.to(device)
print('Model loaded to `%s`'%device)




gpt2_classificaiton_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer, 
                                                          labels_encoder=labels_ids, 
                                                          max_sequence_len=max_length)
print('Dealing with Train...')

train_dataset = ReadKG(path='./data_for_GPT2/WN11/train', 
                               use_tokenizer=tokenizer)
print('Created `train_dataset` with %d examples!'%len(train_dataset))


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=gpt2_classificaiton_collator)
print('Created `train_dataloader` with %d batches!'%len(train_dataloader))

print()

print('Dealing with Validation...')
# Create pytorch dataset.
valid_dataset =  ReadKG(path='./data_for_GPT2/WN11/dev', 
                               use_tokenizer=tokenizer)
print('Created `valid_dataset` with %d examples!'%len(valid_dataset))

# Move pytorch dataset into dataloader.
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=gpt2_classificaiton_collator)
print('Created `eval_dataloader` with %d batches!'%len(valid_dataloader))


print('Dealing with Test...')

test_dataset =  ReadKG(path='./data_for_GPT2/WN11/test', 
                               use_tokenizer=tokenizer)
print('Created `test_dataset` with %d examples!'%len(test_dataset))


test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=gpt2_classificaiton_collator)
print('Created `eval_dataloader` with %d batches!'%len(test_dataloader))





"""## **Train**"""

optimizer = AdamW(model.parameters(),
                  lr = 5e-5, # default is 5e-5
                  eps = 1e-8 # default is 1e-8.
                  )
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

all_loss = {'train_loss':[], 'val_loss':[]}
all_acc = {'train_acc':[], 'val_acc':[]}
print('Epoch')
for epoch in range(epochs):
  print()
  print('Training on batches...')
  train_labels, train_predict, train_loss = train(train_dataloader, optimizer, scheduler, device)
  train_acc = accuracy_score(train_labels, train_predict)
  print('Validation on batches...')
  valid_labels, valid_predict, val_loss = validation(valid_dataloader, device)
  val_acc = accuracy_score(valid_labels, valid_predict)
  print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f"%(train_loss, val_loss, train_acc, val_acc))
  print()
  all_loss['train_loss'].append(train_loss)
  all_loss['val_loss'].append(val_loss)
  all_acc['train_acc'].append(train_acc)
  all_acc['val_acc'].append(val_acc)
#plot_dict(all_loss, use_xlabel='Epochs', use_ylabel='Value', use_linestyles=['-', '--'])
#plot_dict(all_acc, use_xlabel='Epochs', use_ylabel='Value', use_linestyles=['-', '--'])
true_labels, predictions_labels, avg_epoch_loss = validation(test_dataloader, device)
test_acc = accuracy_score(true_labels, predictions_labels)

print("  test_loss: %.5f - test_acc: %.5f "%(avg_epoch_loss, test_acc))
print()
print(true_labels,predictions_labels)

evaluation_report = classification_report(true_labels, predictions_labels, labels=list(labels_ids.values()), target_names=list(labels_ids.keys()))
print(evaluation_report)

# Plot confusion matrix.
plot_confusion_matrix(y_true=true_labels, y_pred=predictions_labels, 
                      classes=list(labels_ids.keys()), normalize=True, 
                      magnify=0.1,
                      );

