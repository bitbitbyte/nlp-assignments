# 课程作业-lesson 02 

## 1. Re-code the house price machine learning

1. Random choose Method to get optimal k* and *b
2. Supervised Direcion to get optimal k* and *b
3. Gradien Descent to get optimal k* and *b
4. Try different Loss function and learning rate.


```python
from sklearn.datasets import load_boston
from matplotlib import pyplot as plt
import random


# 1. Random choose Method to get optimal k* and *b

data = load_boston()
X, y = data['data'], data['target']
X_rm = X[:, 5]

def draw_rm_and_price():
    plt.scatter(X_rm, y)
    plt.show()

def price(rm, k, b):
    """f(x) = k * x + b"""
    return k * rm + b

def loss(y, y_hat):
    return sum((i - j)**2 for i, j in zip(list(y), list(y_hat))) / len(y)

def get_optimal_k_and_b_by_random():
    trying_times = 2000

    min_loss = float('inf')
    best_k, best_b = None, None

    for i in range(trying_times):
        k = random.random() * 200 - 100
        b = random.random() * 200 - 100
        price_by_random_k_and_b = [price(r, k, b) for r in X_rm]

        current_loss = loss(y, price_by_random_k_and_b)

        if current_loss < min_loss:
            min_loss = current_loss
            best_k, best_b = k, b
            print("When time is : {}, get best_k: {} best_b: {}, and the loss is: {}"
                  "".format(i, best_k, best_b, min_loss))

    price_by_fitting = [price(r, best_k, best_b) for r in X_rm]
    plt.scatter(X_rm, price_by_fitting)
    plt.scatter(X_rm, y)
    plt.show()

# 2. Supervised Diretcion to get optimal k* and *b
def get_optimal_k_and_b_by_supervised_direction():

    trying_times = 2000
    up_times = 0

    min_loss = float('inf')

    best_k = random.random() * 200 - 100
    best_b = random.random() * 200 - 100

    direction = [(+1, -1),(-1, +1), (+1, +1), (-1, -1)]
    scalar = 0.1
    next_direction = random.choice(direction)

    for i in range(trying_times):

        k_direction, b_direction = next_direction

        current_k = best_k + k_direction * scalar
        current_b = best_b + b_direction * scalar

        price_by_k_and_b = [price(r, current_k, current_b) for r in X_rm]

        current_loss = loss(y, price_by_k_and_b)

        if current_loss < min_loss:
            min_loss = current_loss
            best_k, best_b = current_k, current_b
            up_times += 1

            print("When time is : {}, get best_k: {} best_b: {}, and the loss is: {}"
                  "".format(i, best_k, best_b, min_loss))
        else:
            next_direction = random.choice(direction)

    price_by_fitting = [price(r, best_k, best_b) for r in X_rm]
    plt.scatter(X_rm, price_by_fitting)
    plt.scatter(X_rm, y)
    plt.show()


# 3. Gradien Descent to get optimal k* and *b
# def partial_k(x, y, y_hat):
#     return -2 / len(y) * sum((y_i - y_hat_i) * x_i for x_i, y_i, y_hat_i
#                             in zip(list(x), list(y), list(y_hat)))
#
# def partial_b(y, y_hat):
#     return -2 / len(y) * sum((y_i - y_hat_i) for y_i, y_hat_i
#                             in zip(list(y), list(y_hat)))
#
# def get_optimal_k_and_b_by_partial():
#
#     trying = 2000
#     learning_rate = 1e-04
#     current_k = random.random() * 200 - 100
#     current_b = random.random() * 200 - 100
#
#     for i in range(trying):
#
#         price_by_b_and_k = [price(r, current_k, current_b) for r in X_rm]
#         current_loss = loss(y, price_by_b_and_k)
#
#         current_k = current_k + -1 * learning_rate * partial_k(X_rm, y, price_by_b_and_k)
#         current_b = current_b +  -1 * learning_rate * partial_b(y, price_by_b_and_k)
#         print("When time is : {}, get best_k: {} best_b: {}, and the loss is: {}"
#               "".format(i, current_k, current_b, current_loss))
#
#     plt.scatter(X_rm, price_by_b_and_k)
#     plt.scatter(X_rm, y)
#     plt.show()

def partial_k(x, y, y_hat):
    n = len(y)

    gradient = 0

    for x_i, y_i, y_hat_i in zip(list(x), list(y), list(y_hat)):
        gradient += (y_i - y_hat_i) * x_i

    return -2 / n * gradient


def partial_b(x, y, y_hat):
    n = len(y)

    gradient = 0

    for y_i, y_hat_i in zip(list(y), list(y_hat)):
        gradient += (y_i - y_hat_i)

    return -2 / n * gradient

def get_optimal_k_and_b_by_partial():
    trying_times = 2000

    X, y = data['data'], data['target']

    min_loss = float('inf')

    current_k = random.random() * 200 - 100
    current_b = random.random() * 200 - 100

    learning_rate = 1e-04

    for i in range(trying_times):

        price_by_k_and_b = [price(r, current_k, current_b) for r in X_rm]

        current_loss = loss(y, price_by_k_and_b)

        print('When time is : {}, get best_k: {} best_b: {}, and the loss is: {}'.format(i, current_k, current_b,
                                                                                                 current_loss))

        k_gradient = partial_k(X_rm, y, price_by_k_and_b)

        b_gradient = partial_b(X_rm, y, price_by_k_and_b)

        current_k = current_k + (-1 * k_gradient) * learning_rate

        current_b = current_b + (-1 * b_gradient) * learning_rate

    plt.scatter(X_rm, price_by_k_and_b)
    plt.scatter(X_rm, y)
    plt.show()

if __name__ == '__main__':
   # draw_rm_and_price()
   # get_optimal_k_and_b_by_random()
   # get_optimal_k_and_b_by_supervised_direction()
   # get_optimal_k_and_b_by_partial()
```



## 2. Answer following questions:

1. Why do we need machine learning methods instead of creating a complicated formula?

   现实世界中往往很多问题过于复杂无法通过数学公式来直接描述，但可以利用基于数据的机器学习方法反而能得到合适的结果。

2. What's the disadvatages of **the 1st Random Choosen** methods in our course?

   当它是完全随机地选择参数k、b的变化方向，而不是有选择性地去选择。

3. Is **the 2nd method supervied direction** better than 1st one? What's the disadvantaged of 2nd supervied direction method?

   第二种方法好于第一种方法。

   是第二种方法假设条件是如果参数k、b在某个方向上变化时loss函数减小，则下一步更新参数时，仍然保持上一个参数变化的方向。它缺点是：

   - 没法保证下一次更新一定使loss函数减小
   - 同时在loss函数是增大的情况下，重新选择参数的变化方向时，依然是采用随机的方式选择参数下一次变化的方向

4. Why do we use **Derivative / Gredient** to fit a target function?

   能够提高拟合的速度

5. In the words 'Gredient Descent', what's the **Gredient** and what's the **Descent**?

   - 梯度是一个方向，沿该方向函数的方向导数最大。
   - 使损失函数不断下降

6. What's the advantages of **the 3rd gradient descent method** compared to the previous methods?

   ​	保证每次参数的更新都使损失函数减小

7. Using the simple words to describe: What's the machine leanring.

   机器学习就是通过数据训练模型，最终使训练得到的模型能够根据输入预测出输出。

## 3. Finish the search problem

Please using the search policy to implement an agent. This agent receives two input, one is @param start station and the other is @param destination. Your agent should give the optimal route based on Beijing Subway system. 

![](E:\图片\ImageMarkdown\timg.jpg)

##### 1.	Get data from web page.

> a.	Get web page source from: https://baike.baidu.com/item/%E5%8C%97%E4%BA%AC%E5%9C%B0%E9%93%81/408485

> b.	You may need @package **requests**[https://2.python-requests.org/en/master/] page to get the response via url

> c.	You may need save the page source to file system.

> d.	The target of this step is to get station information of all the subway lines;

> e.	You may need install @package beautiful soup[https://www.crummy.com/software/BeautifulSoup/bs4/doc/]  to get the url information, or just use > Regular Expression to get the url.  Our recommendation is that **using the Regular Expression and BeautiflSoup both**. 

> f.	You may need BFS to get all the related page url from one url. 
> Question: Why do we use BFS to traverse web page (or someone said, build a web spider)?  Can DFS do this job? which is better? 

##### 2.	Preprocessing data from page source.

> a.	Based on the page source gotten from url. You may need some more preprocessing of the page. 

> b.	the Regular Expression you may need to process the text information.

> c.	You may need @package networkx, @package matplotlib to visualize data. 

> d.	You should build a dictionary or graph which could represent the connection information of Beijing subway routes. 

> e.	You may need the defaultdict, set data structures to implement this procedure. 

##### 3. Build the search agent

> Build the search agent based on the graph we build.

for example, when you run: 

```python
>>> search('奥体中心', '天安门') 
```
you need get the result: 

奥体中心-> A -> B -> C -> ... -> 天安门

- 源码

  ```python
  import requests
  import re
  import os
  from collections import defaultdict
  
  class BeijingDitieSpider:
  
      def __init__(self):
          self.url = 'https://www.bjsubway.com/station/zjgls/#'
  
      def parse_url(self):
          response = requests.get(self.url, verify = False)
          response.encoding = 'gbk'  # 原始网页编码错误，utf-8也不管用，只能用gbk
          return response.text
  
      def get_line_info(self, html):
          """
          返回一个字典，保存地铁线站点信息：{"1号线": [(
          """
          line_info= {}
          pattern = re.compile(r"<td colspan=\"\w\">\w+相邻站间距信息统计表.*?</div>", re.DOTALL)  # 提取不同线路的相邻站间距信息统计表
          for block_html in re.findall(pattern, html):
              block_html = re.sub("<.+?>", "", block_html)  # 删除标签
              each_line_name = re.search(r"(\w+)相邻站间距信息统计表", block_html).group(1)
              two_stations_link = re.findall(r"\s*(\w+——\w+)\s*\n", block_html)  # 提取相邻站点
              two_stations_distance =  re.findall(r"\s*(\d+)\s*\n", block_html)  # 提取相邻站点的距离
              if len(two_stations_distance) != len(two_stations_link):
                  print("%s连接与链接距离不匹配" % each_line_name)
              line_info[each_line_name] = [(link, distance) for link,distance in zip(two_stations_link, two_stations_distance)]
          return line_info
  
      def save_line_graph(self, line_info):
          os.makedirs('./线路图/', exist_ok=True)
          line_names = line_info.keys()
          with open('./线路图/beijing_subline.txt', 'w') as f:
              for name in line_names:
                  f.write(name + '\n')
                  for line in  line_info[name]:
                      f.write(str(line)+ '\n')
                  f.write('\n')
  
      def run(self):
          # 1. 发送请求获取响应
          html = self.parse_url()
          # 2. 获取地铁各线路名
          line_info = self.get_line_info(html)
          # 3. 保存各线路站点信息
          self.save_line_graph(line_info)
  
  class DataPreprocess:
  
      def __init__(self, data_path):
          self.data_path = data_path
          self.line_names = []
          self.line_stations_connects = defaultdict(list)
          self.two_staoins_distance = defaultdict(dict)
          self.stations_connects = defaultdict(str)
          self.station_graph = defaultdict(set)
  
      def load_data(self):
          self.two_staoins_distance = defaultdict(dict)
          two_city_distance = defaultdict(str)
          stations_connetion = []
          with open('./线路图/beijing_subline.txt', 'r') as f:
              for line in f:
                  if not line.strip():
                      self.two_staoins_distance[self.line_names[-1]] = two_city_distance
                      self.line_stations_connects[self.line_names[-1]] = stations_connetion
                      two_city_distance = defaultdict(str)
                      stations_connetion = []
                      continue
                  line = re.sub(r'\,', 'is', line)
                  line = re.sub(r'——', 'to', line)
                  line = re.sub(r'\W', '', line)
                  line = re.split('is', line)
                  if len(line) == 1:
                      self.line_names += line
                      continue
                  two_city_distance[line[0]] = line[1]
                  statoin_near =tuple(line[0].split('to'))
                  stations_connetion.append(statoin_near)
  
      def get_two_stations_distance(self, station1, station2):
  
          for key in self.line_stations_connects:
              self.stations_connects = {**self.stations_connects, ** self.two_staoins_distance[key]}
          self.stations_connects = defaultdict(str, self.stations_connects)
  
          two_station1 = station1 + 'to' + station2
          two_station2 = station2 + 'to' + station1
  
          value1 = self.stations_connects[two_station1]
          value2 = self.stations_connects[two_station2]
          if value1 != '':
              return float(value1)
          elif value2 != '':
              return float(value2)
          else:
              print("这两个站不相邻")
  
      def construction_path(self):
          connets_list = []
          for key in self.line_stations_connects:
              connets_list +=  self.line_stations_connects[key]
          for two_stations in connets_list:
              self.station_graph[two_stations[0]].add(two_stations[1])
              self.station_graph[two_stations[1]].add(two_stations[0])
  
      def run(self):
          self.load_data()
          self.construction_path()
  
  class PathSearch:
  
      def __init__(self, stations_graph, get_two_stations_distance):
          self.stations_graph = stations_graph
          self.get_two_stations_distance = get_two_stations_distance
  
      def search(self, start, distination, search_strategy = lambda n: n ):
  
          paths = [[start]]
          seen = set()
  
          while paths:
              path = paths.pop(0)
              frontier = path[-1]
              if frontier in seen: continue
  
              successors = self.stations_graph[frontier]
  
              for station in successors:
                  if station in path: continue
                  new_path = path + [station]
                  if station == distination: return new_path
                  paths.append(new_path)
  
              seen.add(frontier)
              paths = search_strategy(paths)
  
      def stations_first(self, pathes):
          return sorted(pathes, key=len)
  
      def shortest_first(self, pathes):
          if len(pathes) <= 1: return pathes
  
          def get_path_distance(path):
              distance = 0
              for i in range(len(path)-1):
                  distance += self.get_two_stations_distance(path[i], path[i+1])
              return distance
          return sorted(pathes, key = get_path_distance)
  
  
  if __name__ == '__main__':
      bj_spider = BeijingDitieSpider()
      bj_spider.run()
      
      dp = DataPreprocess('./线路图/beijing_subline.txt')
      dp.run()
  
      ph = PathSearch(dp.station_graph, dp.get_two_stations_distance)
      search_result = ph.search('苹果园', '魏公村', ph.shortest_first)
      print(search_result)
  ```

  

  