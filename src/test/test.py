import random
import pandas as pd
import numpy as np


Q_ij = pd.read_excel(r'F:\File\比赛\物流\文档\py\长株潭供应点-候选地址-货运站数据.xlsx')['供应点年产量/万吨']
d_ij = np.array(pd.read_excel(r'F:\File\比赛\物流\文档\py\供应点-候选地址距离矩阵.xls'))
d_jk = np.array(pd.read_excel(r'F:\File\比赛\物流\文档\py\候选地址-货运站距离矩阵.xls'))
f_j = 8e8    #建设在候选地址j建设铁路物流中心的成本费
C = 0.35     #单位运输费用，单位：元/吨公里
P = 3        #在10个候选地址中选择3个建设铁路物流中心
M = range(0,55)       #55个供应点
N = range(0,10)       #10个候选地址
L = range(0,16)      #16个货运站
U_L = 300            #货运站的最大容量

#def function(P,f_j,C,Q_ij,d_ij,d_jk，M,N,L):
f1 = 3 * f_j #建设P个铁路物流中心成本费用

f2 = 0
log_center = random.sample(range(0, 10), P)#在10个候选地址中随机选择P个做为铁路物流中心
print(log_center)
i_j = []
for i in M:
    i_j.append(random.choice(log_center)) #供应点i货物运往随机选择的P个铁路物流中心之一
print(i_j)
Q_p = [0,0,0]  #3个铁路物流中心分别接受的总货运量
for i in M:
    f2 += C * d_ij[i,i_j[i]] * Q_ij[i]  #所有货物从供应点至铁路物流中心的总运输费用
    if i_j[i] == log_center[0]:
        Q_p[0] += Q_ij[i]
    elif i_j[i] == log_center[1]:
        Q_p[1] += Q_ij[i]
    elif i_j[i] == log_center[2]:
        Q_p[2] += Q_ij[i]
print(f2)
print(Q_p)

f3 = 0
#j_k = []
#for k in L:
#    j_k.append(random.choice(log_center)) #供应点i货物运往随机选择的P个铁路物流中心之一
#print(i_j)
new_d_jk = d_jk[:,log_center]

print(new_d_jk)

mat_volume = np.full((16,1),U_L)  #每个货运站可用容量
mat_trans = np.zeros((16,3))
l = 0
r = 0
print(mat_volume)
print(mat_trans)

# print(new_d_jk[1][1])
mat_volume = np.full((16,1),U_L)  #每个货运站可用容量
mat_trans = np.zeros((16,3))
l = 0
r = 0
while Q_p != [0,0,0]:   #当铁路物流中心所有货物全部运到货运站时停止

    min_ans = 1652130000   #设一个非常大的数
    for i in  L :
        for j in range(P):
             if new_d_jk[i][j] < min_ans:
                min_ans = new_d_jk[i][j]
                r = i
                l = j
    #x = min_ans,r,l   #求铁路物流中心与货运站距离最小的组合
    new_d_jk[r][l] = 1652130000
    print(Q_p)
    ans = Q_p[l]
    if Q_p[l] > mat_volume[r]:
        Q_p[l] = ans- mat_volume[r]            #铁路物流中心l的货物运往货运站r
        mat_trans[r][l] = mat_volume[r]    #记录铁路物流中心l向货运站r运输量
        mat_volume[r] = 0                  #运满货运站r
    else:
        mat_trans[r][l] = ans    #记录铁路物流中心l向货运站r运输量
        mat_volume[r] -= ans           #货运站r剩余部分容量
        Q_p[l] = 0
        #铁路物流中心l的货物运往货运站r
print("################")
print(mat_volume)
print(mat_trans)