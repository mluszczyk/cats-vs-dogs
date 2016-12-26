import lib

data = lib.load_data()
train_X = data[0]
shape = train_X.shape[1:]

model = lib.get_convnet_model(shape)

lib.interactive_train_convnet(model, data)
