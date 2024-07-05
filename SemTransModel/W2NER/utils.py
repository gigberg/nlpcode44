import logging
import pickle
import time
from collections import defaultdict, deque


def get_logger(dataset):
    pathname = "./log/{}_{}.txt".format(dataset, time.strftime("%m-%d_%H-%M-%S"))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(pathname)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def save_file(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_file(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def convert_index_to_text(index, type):
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type)
    return text


def convert_text_to_index(text): # 与上一个相反
    index, type = text.split("-#-")
    index = [int(x) for x in index.split("-")]
    return index, int(type)

# #entities: [index_list, int(type),,,,,]
# def decode(outputs, entities, length): # outputs, entity_text([index_list_text('-') + "-#-" + ent_typeid_text,,,,]), sent_length
#     class Node: # citod 邓凯方图卷积中也是定义了一个node对象来记录同一个位置的多种属性(如果用多个list的相同的index来隐含的位置对应关系yexk,但容易操作不当出错,且不易理解)
#         def __init__(self): #每个节点的信息, 每个节点(sent中的每个word)既可以作为head(记录它对应的tail,,作为head节点时也要[看成中间节点并记录后一部分的信息]),也可以作为中间节点(记录它的next_index[超过tail则截断]和所在head_tail)
#             self.THW = []                # [(tail, type)]
#             self.NNW = defaultdict(set)   # {(head,tail): {next_index}}

#     ent_r, ent_p, ent_c = 0, 0, 0
#     decode_entities = []
#     q = deque() #被傻瓜改成非递归了,难懂的要死, 还是李京烨的递归源码容易理解
#     for instance, ent_set, l in zip(outputs, entities, length): #展开batch(类似rnn展开batch,rnn双层for循环),,单条sample来解码 # outputs, entity_text([index_list_text('-') + "-#-" + ent_typeid_text,,,,]), sent_length
#         predicts = [] # 单个ent保存为([index_list], ent_type_id)格式
#         nodes = [Node() for _ in range(l)] #l是一句话的有效长度(不含pad/cls/sep)
#         for cur in reversed(range(l)): # 深度优先搜索, cur是list类型,保存当前ent的index_list,,,,,,先全n*n遍历,找到所有tail为cur_i的head, 并记录该head
#             heads = [] # len=3, cur in [2,1,0]
#             for pre in range(cur+1): #pre(含cur), 下三角THW(含对角线)Tail_i->Head_j
#                 # THW,,,这层(下三角遍历[非深度搜索]用于发现所有头节点,并记录它的tail相关信息
#                 # # cur行, pre 列, 目前是下三角含对角线, 并且是从n到1逐渐缩小的下三角
#                 if instance[cur, pre] > 1: # batch内单个sample平面n*n, 大于1说明是NHW
#                     nodes[pre].THW.append((cur, instance[cur, pre])) # head节点, 添加(tail,type)
#                     heads.append(pre) #heads列表, 暂存所有待遍历的start节点, 一个head可能有多个tail(也即有多个entity共用head)
#                 # NNW #(每一步迭代同时还需要处理)上三角(注意逆转了i,j的位置), 且非对角线位置
#                 if pre < cur and instance[pre, cur] == 1: #情况一:两元素实体,先记录下来,怎么不出力pre=cur的单元素实体情况??
#                     # cur node
#                     for head in heads: # (每一步迭代同时还需要处理)
#                         nodes[pre].NNW[(head,cur)].add(cur) #超复杂的dict index: (head,cur)
#                     # post nodes(tail node)
#                     for head,tail in nodes[cur].NNW.keys(): # (每一步迭代同时还需要处理)
#                         if tail >= cur and head <= pre:
#                             nodes[pre].NNW[(head,tail)].add(cur)
#             # entity #  #情况二:多元素实体
#             for tail,type_id in nodes[cur].THW: #已知所有cur开头的实体的tail[也即下三角中列为cur的底部(类似贪心搜索,每次都是最优,)
#                 if cur == tail: # 对角线上 单元素实体
#                     predicts.append(([cur], type_id))
#                     continue
#                 q.clear()
#                 q.append([cur]) #head=cur
#                 while len(q) > 0: # 双端队列实现深度优先搜索
#                     chains = q.pop()
#                     for idx in nodes[chains[-1]].NNW[(cur,tail)]: # 还已知所有cur-tail的中间节点
#                         if idx == tail:
#                             predicts.append((chains + [idx], type_id))
#                         else:
#                             q.append(chains + [idx])

#         predicts = set([convert_index_to_text(x[0], x[1]) for x in predicts])
#         decode_entities.append([convert_text_to_index(x) for x in predicts])
#         ent_r += len(ent_set) # 真假真负例真值表的样本数, c是 左上角, p是左侧p的分母, r是上册r的分母 # 真实正例
#         ent_p += len(predicts) # 预测正例
#         ent_c += len(predicts.intersection(ent_set)) #集合交集, 真正例(分子)
#     return ent_c, ent_p, ent_r, decode_entities



# 原版递归版decode
def decode(outputs, entities, length):
    ent_r, ent_p, ent_c = 0, 0, 0
    decode_entities = [] # # instance[n*n], ent_set[list of index+list+type_id+str], l(sent len[no cls/sep/pad])
    for index, (instance, ent_set, l) in enumerate(zip(outputs, entities, length)):
        forward_dict = {} # 所有word各自的可能的后驱列表(含tail)
        head_dict = {} # 所有word各自的可能的tail列表
        ht_type_dict = {} #所有下三角区域的位置各自的(实体词)type列表
        for i in range(l):
            for j in range(i + 1, l): #上三角,后对前的影响,找后驱(并且已知后驱只在i后边部分三角)
                if instance[i, j] == 1:
                    if i not in forward_dict: # 先初始化list
                        forward_dict[i] = [j]
                    else:
                        forward_dict[i].append(j)
        for i in range(l): #下三角,前对后的影响,找head[并且已知前驱只在i前边部分三角(两个三角也即论文中说的cln direction的原因)
            for j in range(i, l): # 下三角j in range(i, l)
                if instance[j, i] > 1: # 下三角instance[j, i],,,instance[j, i] > 1
                    ht_type_dict[(i, j)] = instance[j, i] # n*n的二元嵌套dict透过改变index结构,变成n*n的一元faltten的dict
                    if i not in head_dict:
                        head_dict[i] = {j}
                    else:
                        head_dict[i].add(j)
        #上边已经全图节点循环了一遍,记录了全图节点(上三角ht_type_dict)/全行(上三角forward_dict)/全列(下三角head_dict)的待用信息,,,用于图深度优先搜索(上三角,或者脱离n*n平面来看,,直接用上三角forward_dict来建图的[也即【1.】不是用邻接矩阵表示图,而是类似依存树用dict+后驱列表来表示图(也即【2.】链表法表示图,类似hash的链表法,,所谓链表法用的是value=tails_list的【dict】,但也可以用list of object来表示,也即每个链表看成一个【类对象/结构体】,唯一id属性是head,存储信息的属性是tails列表)])
        predicts = []

        def find_entity(key, entity, tails):
            entity.append(key)
            if key not in forward_dict: #该head不存在后驱?
                if key in tails: # 如果是单个词构成的实体
                    predicts.append(entity.copy())
                entity.pop() # 深度优先的回溯状态还原
                return #否则啥也不做
            else:
                if key in tails: # #该head存在后驱,但是也存在单个词构成的实体?
                    predicts.append(entity.copy()) #append浅拷贝(后者用slice[:],slice当右值时y =x[:]此时是浅拷贝(实测不是深拷贝哦),但是slice当左值时,x[:1]=[999],或者x[0]=999这个是inplace赋值),避免随时变动的entity(path)影响predict中此前保存的有效路径值
            for k in forward_dict[key]:
                find_entity(k, entity, tails) # k:head,,,entity(已经搜索的path[注意搜索建图时候只沿着forward_dict所有word各自的可能的后驱列表(含tail)建图]),,,tail所有word各自的可能的tail列表
            entity.pop()  # 深度优先的回溯状态还原

        for head in head_dict:# # 所有word的可能的tail列表
            find_entity(head, [], head_dict[head]) #第二个参数[]是保存的搜索路径(ent mention index list), 并且在搜索到tail(head_dict[head]))时候append到全局变量predicts=[]中

        predicts = set([convert_index_to_text(x, ht_type_dict[(x[0], x[-1])]) for x in predicts]) #去重？深度优先不该出现重复啊? [[0], [1], [2]] - > ['0-#-2', '1-#-2', '2-#-2']
        decode_entities.append([convert_text_to_index(x) for x in predicts]) #原先的predict不含entity_type信息, 上一步convert_index_to_text是为了添加type信息
        ent_r += len(ent_set) #list[enty_index_text_type_text],全真实值样本数/分母
        ent_p += len(predicts) #全部预测值样本数/分母
        for x in predicts:
            if x in ent_set:
                ent_c += 1
    return ent_c, ent_p, ent_r, decode_entities


def cal_f1(c, p, r): # total_ent_c, total_ent_p, total_ent_r,真假真负例真值表的样本数(all batches)
    if r == 0 or p == 0:  #样本统计发现全是真负例的情形
        return 0, 0, 0

    r = c / r if r else 0
    p = c / p if p else 0

    if r and p:
        return 2 * p * r / (p + r), p, r
    return 0, p, r
