import torch
from fastprogress import master_bar, progress_bar
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# para poder usar la gpu de nvidia
device = "cuda" if torch.cuda.is_available() else "cpu"

class MyModel(torch.nn.Module):
  def __init__(self, layers):
    super().__init__()
    self.layers = torch.nn.ModuleList(layers)

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

class MySequentialModel():
  def __init__(self, layers):
    self.net = MyModel(layers)

  def compile(self, loss, optimizer, metrics):
    self.loss = loss
    self.optimizer = optimizer
    self.metrics = metrics
    

  def fit(self, X_train, y_train, epochs, validation_data, batch_size=32):
    self.net.to(device)

    X_train, y_train = torch.from_numpy(X_train).float(), torch.from_numpy(y_train)
    X_valid, y_valid = validation_data
    X_valid, y_valid = torch.from_numpy(X_valid).float(), torch.from_numpy(y_valid)

    self.history = {"train_loss": [], "val_loss": []}
    for metric in self.metrics:
        self.history[f'train_{metric.name}'] = []
        self.history[f'val_{metric.name}'] = []

    mb = master_bar(range(1, epochs+1))
    for epoch in mb:
      # entrenamiento
      self.net.train()
      # inicializamos métricas y función de pérdida
      train_loss, train_metrics = [], [[] for m in self.metrics]
      # iteramos sobre los datos
      for i in progress_bar(range(0, len(X_train), batch_size), parent=mb):
        # extraemos los datos
        X, y = X_train[i:i+batch_size], y_train[i:i+batch_size]
        # y los enviamos a la gpu
        X, y = X.to(device), y.to(device)
        # ponemos a zero los gradientes del optimizador
        self.optimizer.zero_grad()
        # llamamos a la red neuronal con los inputs para que devuelva los outputs
        output = self.net(X)
        # y lo usamos para calcular la función de pérdida
        loss = self.loss(output, y)
        # calculamos los gradientes
        loss.backward()
        # actualizamos los pesos
        self.optimizer.step()
        # guardamos la función de pérdida
        train_loss.append(loss.item())
        # comentario para la barra de progreso
        comment = f'train_loss {np.mean(train_loss):.5f}'
        # guardamos todas las métricas
        for i, metric in enumerate(self.metrics):
          train_metrics[i].append(metric.call(output, y))
          comment += f' train_{metric.name} {np.mean(train_metrics[i]):.5f}'
        mb.child.comment = comment
    # evaluación
    self.net.eval()
    # instanciamos métricas y función de pérdida
    val_loss, val_metrics = [], [[] for m in self.metrics]
    # para no calcular gradiente
    with torch.no_grad():
      # iteramos por todos los datos
      for i in progress_bar(range(0, len(X_valid), batch_size), parent=mb):
        # extraemos datos
        X, y = X_valid[i:i+batch_size], y_valid[i:i+batch_size]
        # y los enviamos a la gpu
        X, y = X.to(device), y.to(device)
        # calculamos los outputs del modelo
        output = self.net(X)
        # calculamos la función de pérdida
        loss = self.loss(output, y)
        val_loss.append(loss.item())
        comment = f'val_loss {np.mean(train_loss):.5f}'
        for i, metric in enumerate(self.metrics):
          val_metrics[i].append(metric.call(output, y))
          comment += f' val_{metric.name} {np.mean(train_metrics[i]):.5f}'
        mb.child.comment = comment
    
        # guardamos métricas
        self.history["train_loss"].append(np.mean(train_loss))
        self.history["val_loss"].append(np.mean(val_loss))
        for i, metric in enumerate(self.metrics):
            self.history[f'train_{metric.name}'].append(np.mean(train_metrics[i]))
        self.history[f'val_{metric.name}'].append(np.mean(val_metrics[i]))
        # barra de progreso
        comment = f'Epoch {epoch}/{epochs} train_loss {np.mean(train_loss):.5f} val_loss {np.mean(train_loss):.5f}'
        for i, metric in enumerate(self.metrics):
            comment += f' train_{metric.name} {np.mean(train_metrics[i]):.5f} val_{metric.name} {np.mean(val_metrics[i]):.5f}'
        mb.write(comment)

    return self.history

  def predict(self, X_new):
    # enviamos el modelo a la gpu
    self.net.to(device)
    # ponemos el modelo en evaluación
    self.net.eval()
    # convertir los datos a un tensor de pytorch
    X = torch.from_numpy(X_new).float()
    # calculamos la salida del modelo usando softmax para convertirlo en una distribución de probabilidad
    return torch.softmax(self.net(X.to(device)), axis=1)

  def evaluate(self, X_test, y_test, batch_size=32):
    # enviamos el modelo a la gpu
    self.net.to(device)
    # ponemos el modelo en modo evaluación
    self.net.eval()
    # convertimos los datos en tensores
    X_test, y_test = torch.from_numpy(X_test).float(), torch.from_numpy(y_test)
    mb = master_bar(range(0, 1))
    # bucle para evaluar métricas
    for e in mb:
      total_loss, metrics = [], [[] for m in self.metrics]
      # para no calcular gradientes
      with torch.no_grad():
        # iteramos por todos los datos
        for i in progress_bar(range(0, len(X_test), batch_size), parent=mb):
          X, y = X_test[i:i+batch_size], y_test[i:i+batch_size]
          X, y = X.to(device), y.to(device)
          output = self.net(X)
          loss = self.loss(output, y)
          total_loss.append(loss.item())
          comment = f'loss {np.mean(total_loss):.5f}'
          for i, metric in enumerate(self.metrics):
            metrics[i].append(metric.call(output, y))
            comment += f' val_{metric.name} {np.mean(metrics[i]):.5f}'
          mb.child.comment = comment
        mb.write(comment)



class Accuracy():
  def __init__(self):
    self.name = "acc"
  
  def call(self, output, labels):
    return (torch.argmax(output, axis=1) == labels).sum().item() / labels.shape[0] 



X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

X, y = X.values / 255., y.values.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)


model = MySequentialModel([
  torch.nn.Linear(784, 100),
  torch.nn.ReLU(),
  torch.nn.Linear(100, 10)    
])

model.compile(loss=torch.nn.CrossEntropyLoss(),
              optimizer=torch.optim.SGD(model.net.parameters(), lr=0.1),
              metrics=[Accuracy()])

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))


pd.DataFrame(history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()


model.evaluate(X_test, y_test)

X_new = X_test[:10]
y_proba = model.predict(X_new)
y_proba.round()


plt.figure(figsize=(7.2, 2.4))
y_pred = torch.argmax(y_proba, axis=1)
for index, image in enumerate(X_new):
    plt.subplot(2, 5, index + 1)
    plt.imshow(image.reshape(28,28), cmap="binary", interpolation="nearest")
    plt.axis('off')
    plt.title(f'{y_pred[index].item()}/{y_test[index]}', fontsize=12, color="green" if y_pred[index].item() == y_test[index] else "red")
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()