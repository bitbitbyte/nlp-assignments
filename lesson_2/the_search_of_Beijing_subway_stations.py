
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
        os.makedirs('./Beijing_Subway_map_data/', exist_ok=True)
        line_names = line_info.keys()
        with open('./Beijing_Subway_map_data/beijing_subline.txt', 'w') as f:
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
        with open('./Beijing_Subway_map_data/beijing_subline.txt', 'r') as f:
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

    def pretty_print(self, station_list):

        print_str = ''
        for i in range(len(station_list)-1):
            print_str += station_list[i] + '=>'
        print_str += station_list[-1]
        print(print_str)



if __name__ == '__main__':
    bj_spider = BeijingDitieSpider()
    bj_spider.run()

    dp = DataPreprocess('./Beijing_Subway_map_data/beijing_subline.txt')
    dp.run()

    ph = PathSearch(dp.station_graph, dp.get_two_stations_distance)
    search_result = ph.search('西局', '天安门西', ph.shortest_first)
    ph.pretty_print(search_result)