def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]

def test_load():
  return 'loaded'

def cond_probs_product(table, e_val, t_col, t_val):
  cond_prob_list =[]
  table_colums = up_list_column_names(table)
  evidence_columns = table_columns[:-1]
  evidence_columns
  evidence_complete = up_zip_lists(evidence_columns, e_val)
  
def naive_bayes(table, evidence_row, target):
  #compute P(Flu=0|...) by collecting cond_probs in a list, take the produce of the list, finally multiply by P(Flu=0)
  p_num = cond_probs_product(table, evidence_row, target, 0)
  p_a = prior_prob(table, target, 0)
  neg = p_num * p_a


  #do same for P(Flu=1|...)
  p_num = cond_probs_product(table, evidence_row, target, 1)
  p_a = prior_prob(table, target, 1)
  pos = p_num * p_a

  #Use compute_probs to get 2 probabilities
  prediction = compute_probs(neg,pos)
  
  #return your 2 results in a list
  return prediction

def prior_prob(table, t_col, t_val):
  t_list = up_get_column(table, t_col)
  p_a = sum([1 if v==t_val else 0 for v in t_list])/len(t_list)
  return p_a

def naive_bayes(table, evidence_row, target):
  #compute P(Flu=0|...) by collecting cond_probs in a list, take the produce of the list, finally multiply by P(Flu=0)
  p_num = cond_probs_product(table, evidence_row, target, 0)
  p_a = prior_prob(table, target, 0)
  neg = p_num * p_a
  
def cond_prob(table, evidence, evidence_value, target, target_value):
  t_subset = up_table_subset(table, target, 'equals', target_value)
  e_list = up_get_column(t_subset, evidence)
  p_b_a = sum([1 if v==evidence_value else 0 for v in e_list])/len(e_list)
  return p_b_a


  #do same for P(Flu=1|...)
  p_num = cond_probs_product(table, evidence_row, target, 1)
  p_a = prior_prob(table, target, 1)
  pos = p_num * p_a

  #Use compute_probs to get 2 probabilities
  prediction = compute_probs(neg,pos)
  
  #return your 2 results in a list
  return prediction

def metrics(a_list):
  assert isinstance(a_list, list), f'Parameter must be a list'
  for item in a_list:
    assert isinstance(item, list), f'Parameter must be a list of lists'
    assert len(item) == 2, f'Parameter must be a zipped list'
  
      
  TN = sum([1 if pair==[0,0] else 0 for pair in a_list])
  TP = sum([1 if pair==[1,1] else 0 for pair in a_list])
  FP = sum([1 if pair==[1,0] else 0 for pair in a_list])
  FN = sum([1 if pair==[0,1] else 0 for pair in a_list])

  accuracy = sum([p==a for p, a in a_list])/len(a_list)
  precision = TP/(TP+FP) if TP+FP> 0 else 0
  recall = TP/(TP+FN) if TP+FN> 0 else 0
  f1 = 2*precision * recall/(precision + recall) if precision + recall> 0 else 0
  return {f'Precision': precision, 'Recall': recall, 'F1': f1, 'Accuracy': accuracy}

