#학습 알고리즘 구현:SGD(미니배치로 데이터 뽑고 반복)
import numpy as np
import sys, os 
sys.path.append(os.pardir) #부모 디렉토리에서 가지고 올수 있게 설정
from common.functions import * #전부 다 갖고 온다는 뜻?
from common.gradient import numerical_gradient
import matplotlib.pyplot as plt

# def cross_entropy_error(y, t):
#     if y.ndim == 1:
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)
        
#     # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
#     if t.size == y.size:
#         t = t.argmax(axis=1)
             
#     batch_size = y.shape[0]
#     return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01): #weight_init_std?
        self.W=np.random.randn(2,3) 
        self.params={}
        
        self.params['W1']=weight_init_std*\
        np.random.randn(input_size,hidden_size)
        self.params['b1']=np.zeros(hidden_size)

        self.params['W2']=weight_init_std*\
        np.random.randn(hidden_size,output_size)
        self.params['b2']=np.zeros(output_size)        
    
    def predict(self,x):
         W1,W2=self.params['W1'],self.params['W2']
         B1,B2=self.params['b1'],self.params['b2']
         a1=np.dot(x,W1)+B1
         z1=sigmoid(a1)
         a2=np.dot(z1,W2)+B2
         y=softmax(a2)

         return y
     
    def loss(self,x,t): 
         y=self.predict(x)
         loss=cross_entropy_error(y,t)
         
         return loss
    
    
    def accuracy(self,x,t):
         y=self.predict(x)      #ex)y=[[0.1 0.8 0.1],[0.5 0.3 0.2],[0.4 0.3 0.3]]
         y=np.argmax(y,axis=1)  #[1,0,0]
         t=np.argmax(t,axis=1)  #[1,0,0]

         accuracy=np.sum(y==t)/float(x.shape[0]) #x.shape[0]은 입력 데이터의 개수  이게 이해가 안 감
         return accuracy
    
    def numerical_gradient(self,x,t):
         loss_W=lambda W:self.loss(x,t)

         grads={} #다른곳에서 참조할 수 없는 딕셔너리
         grads['W1']=numerical_gradient(loss_W,self.params['W1']) #기존에 있었던 numerical_gradient(f,x)
         grads['b1']=numerical_gradient(loss_W,self.params['b1'])
         grads['W2']=numerical_gradient(loss_W,self.params['W2'])
         grads['b2']=numerical_gradient(loss_W,self.params['b2'])

         return grads
    
# dW=numerical_gradient(f,net.W) #f 함수, x 변수(net.W:2x3의 행렬)를 인수로 받음

#f=lambda w(변수):net.loss(x,t)(함수)
#dw=numerical_gradient(f,net.W)




#first step: Acquire parameter
# net=TwoLayerNet(input_size=784,hidden_size=100,output_size=10) 

# net.params['W1'].shape #784,100
# net.params['b1'].shape #100
# net.params['W2'].shape #100 10
# net.params['b2'].shape #10

#second step: processing 예측

# x=np.random.rand(100,784) #100x784의 더미 데이터,0과 1사이의 값 아무거나
# y=net.predict(x) #100x10
# t=np.random.rand(100,10)

# grads=net.numerical_gradient(x,t)
# grads['W1'].shape
# grads['b1'].shape
# grads['W2'].shape
# grads['b2'].shape



#미니배치 학습 구현하기
import numpy as np
import sys, os 
sys.path.append(os.pardir) 
from dataset.mnist import load_mnist 
import matplotlib.pyplot as plt

(x_train,t_train),(x_test,t_test)=\
        load_mnist(normalize=True,one_hot_label=True) 

train_loss_list=[]

iters_num=10000
train_size=x_train.shape[0] #6만개
batch_size=100
learning_rate=0.1

network=TwoLayerNet(input_size=784,hidden_size=50,output_size=10)

for i in range(iters_num):
     batch_mask=np.random.choice(train_size,batch_size) 
     x_batch=x_train[batch_mask]
     t_batch=t_train[batch_mask] 

     grad=network.numerical_gradient(x_batch,t_batch) #1.기울기 구하기 

     for key in ('W1','b1','W2','b2'): #딕셔너리 참조법
          network.params[key]-=learning_rate*grad[key] #2.경사법으로 매개변수 구하고 갱신하기
     loss =network.loss(x_batch,t_batch) #갱신한 매개변수로 Loss 함수 구하기
     train_loss_list.append(loss)


x=np.arange(0,iters_num)
plt.plot(x,train_loss_list)
plt.show()

# 시험 데이터로 평가하기
import numpy as np
import sys, os 
sys.path.append(os.pardir) 
from dataset.mnist import load_mnist 
import matplotlib.pyplot as plt

(x_train,t_train),(x_test,t_test)=\
        load_mnist(normalize=True,one_hot_label=True) 

network=TwoLayerNet(input_size=784,hidden_size=50,output_size=10)

iters_num=10000
train_size=x_train.shape[0] #6만개
batch_size=100
learning_rate=0.1


train_loss_list=[]
train_acc_list=[]
test_acc_list=[]

iters_per_epoch=max(train_size/batch_size,1) #현재 1에폭=600

for i in range(iters_num):
     batch_mask=np.random.choice(train_size,batch_size) 
     x_batch=x_train[batch_mask]
     t_batch=t_train[batch_mask] 

     grad=network.numerical_gradient(x_batch,t_batch) #1.기울기 구하기 

     for key in ('W1','b1','W2','b2'): #딕셔너리 참조법
          network.params[key]-=learning_rate*grad[key] #2.경사법으로 매개변수 구하고 갱신하기
     loss =network.loss(x_batch,t_batch) #갱신한 매개변수로 Loss 함수 구하기
     train_loss_list.append(loss)

     if i % iters_per_epoch ==0: # 1에폭당 정확도 계산 ex) i가 600이 될때 실행
          train_acc=network.accuracy(x_train,t_train)
          test_acc=network.accuracy(x_test,t_test)
          train_acc_list.append(train_acc)
          test_acc_list.append(test_acc)
          print("train acc, test acc \ "
                +str(train_acc)+", " + str(test_acc))

x=np.arange(0,iters_num)
plt.plot(x,train_loss_list)
plt.show()

markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()