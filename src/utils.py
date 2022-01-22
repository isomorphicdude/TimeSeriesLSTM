def getData(data, target, window, train_split, \
   batch_size=20, horizon=0):
   '''
     Implement custom function for creating 
     datasets for training.
     Parameters: 
        - data: list
        - target: list
        - window: int, the length of x interval
        - train_split: int the number of train samples
        - batch_size: int
        - horizon: int, default 0,the length of prediction wanted
     Output:
        - result: 
   '''
   x_vals = []
   y_vals = []
   start = 0 + window
   end = len(data) - horizon

   for i in range(start, end):
      x = data[(i-window):i,:]
      y = target[i:(i+horizon+1),:]
      x_vals.append(x)
      y_vals.append(y)
        
   tensor_x = torch.Tensor(x_vals)
   tensor_y = torch.Tensor(y_vals)

   set_train = TensorDataset(tensor_x,tensor_y)
   print(set_train.shape)
   loader_train = DataLoader(set_train, batch_size = batch_size,\
      sampler=sampler.SubsetRandomSampler(range(train_split)))
   loader_test = DataLoader(set_train, batch_size = batch_size,\
      sampler=sampler.SubsetRandomSampler(range(train_split, len(set_train))))
   
   return (loader_train,loader_test)