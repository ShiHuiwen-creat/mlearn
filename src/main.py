import argparse
import shutil

from find_solution import *
from gen_matrix import *
from gen_test_case import *


# 上面就是导入一些包

# 打印error信息
def print_error(value):
    print('error:', value)


# 转变
def convert(output, symbol):
    contents = set()  # set集合
    #二层循环
    for i in range(len(output)):
        re = ""
        for j in range(len(output[i])):
            re = re + symbol[j] + '=' + str(output[i][j]) + '\n'
        contents.add(re)  # 添加一些字符串，字符串是用来生成约束条件的
    return contents


def task(bubble_matrix, constraint, bubble_input, idx, number=1):
    output = get_test_cases(bubble_matrix[idx].astype(np.int).copy(), constraint.copy(), bubble_input[idx].copy(),
                            number)
    c = convert(output, bubble_input[idx].copy())
    return c


# 变异，就是查看约束条件
def mutate(constraint, or_cons):
    while True:
        idx = random.randint(0, len(constraint) - 1)
        if (constraint[idx] in or_cons):
            continue
        if constraint[idx].find('>') != -1:
            constraint[idx] = constraint[idx].replace('>', '<')
            return constraint
        if constraint[idx].find('<') != -1:
            constraint[idx] = constraint[idx].replace('<', '>')
            return constraint
        if constraint[idx].find('>=') != -1:
            constraint[idx] = constraint[idx].replace('>=', '<=')
            return constraint
        if constraint[idx].find('<=') != -1:
            constraint[idx] = constraint[idx].replace('<=', '>=')
            return constraint
        if constraint[idx].find('==') != -1:
            constraint[idx] = constraint[idx].replace('==', '!=')
            return constraint
        if constraint[idx].find('!=') != -1:
            constraint[idx] = constraint[idx].replace('!=', '==')
            return constraint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Case Generator')  # 解析器
    parser.add_argument('--name', help='c program name') # 添加参数name是c文件
    parser.add_argument('--n', type=int, default=10000, help='test case number (default: 10000)') # 测试用例的数量
    parser.add_argument('--r', type=float, default=0.5, help='ratio of positive case (default: 0.5)') # 正例的比例
    parser.add_argument('--c', type=int, default=-1, help='thread number (default: number of cores(-1))') # 线程的数量
    arg = parser.parse_args()
    kwargs = vars(arg)  # 返回对象的属性和属性值的字典对象，就和哈希表差不多

    time_start = time.time() # 开始时间

    # parameters
    input_path = 'F:\git\python\\test-case-generator\\resources\\t0.c'; #路径
    NUM = kwargs['n'] # 测试用例的数量
    ratio = kwargs['r'] # 比例
    pos_num = int(NUM * ratio) #正例的数量
    neg_num = NUM - pos_num # 反例的数量
    if kwargs['c'] == -1: #如果线程的数量是-1的话
        core_num = int(mp.cpu_count()) # multiprocessing 模块，cpu_count() 查找cpu的数量
    else:
        core_num = kwargs['c'] # 就等于线程的数量
    output_path = './case' #输出路劲在output

    # clear
    # 下面的一个os的代码块就是如果有case的话就删除，再创建新的case和positive和negative，为了新的input不会和以前的约束和测试用例冲突
    if os.path.exists('./case'):
        shutil.rmtree('./case')
    os.mkdir('./case')
    os.mkdir('./case/positive')
    os.mkdir('./case/negative')

    time_clear = time.time() # 数据清洗的时间
    print("清理完成（{:.3f}s）...".format(time_clear - time_start))

    # init 初始化
    input, constraint = analysis(input_path) # 分析c代码生成
    # 通过分析源代码生成的对象生成
    input, pos_constraint, or_cons = revised(input, constraint)
    with open('./case/constraints.txt', 'w') as f: # 生成一个约束条件集
        for line in pos_constraint:  # 遍历pos_constraints来写在这个文件上
            f.write(line + '\n')
    neg_constraint = mutate(pos_constraint.copy(), or_cons) # 生成neg_constraints文件
    pos_matrix = gen_matrix(input, pos_constraint) # 生成矩阵
    neg_matrix = gen_matrix(input, neg_constraint) # 生成矩阵
    pos_bubble_input, pos_bubble_matrix = matrix_split(pos_matrix, input, pos_constraint) #生成postion测试用例输入喝测试用例矩阵
    neg_bubble_input, neg_bubble_matrix = matrix_split(neg_matrix, input, neg_constraint)#生成negtive测试用例输入喝测试用例矩阵
    pos_n = len(pos_bubble_matrix)#长度
    neg_n = len(neg_bubble_matrix)
    pos_part_num = math.ceil(5 * pos_num ** (1 / pos_n))#上整数
    neg_part_num = math.ceil(5 * neg_num ** (1 / neg_n))
    pos_bubbles = list()
    pos_results = list()
    neg_bubbles = list()
    neg_results = list()
    pool = mp.Pool(core_num) # 线程池

    time_init = time.time()
    print("文件解析及初始化完成（{:.3f}s）...".format(time_init - time_clear))

    # solute
    #生成 结果集
    for i in range(len(pos_bubble_matrix)):
        pos_result = pool.apply_async(task, (pos_bubble_matrix, pos_constraint, pos_bubble_input, i, pos_part_num),
                                      error_callback=print_error)
        pos_results.append(pos_result)
    for i in range(len(neg_bubble_matrix)):
        neg_result = pool.apply_async(task, (neg_bubble_matrix, neg_constraint, neg_bubble_input, i, neg_part_num),
                                      error_callback=print_error)
        neg_results.append(neg_result)

#线程池关闭
    pool.close()
    pool.join()

    for result in pos_results:
        pos_bubbles.append(result.get())
    for result in neg_results:
        neg_bubbles.append(result.get())

    time_solute = time.time()
    print("约束求解完成（{:.3f}s）...".format(time_solute - time_init))

    # merge and generate
    pool = mp.Pool(core_num)
    # 下面两行代码就是主要生成测试用例的
    pos_count = generator(pos_bubbles, output_path + '/positive', pos_num, pool)
    neg_count = generator(neg_bubbles, output_path + '/negative', neg_num, pool)
    pool.close()
    pool.join()

    count = pos_count + neg_count
    time_gen = time.time()
    print("生成测试用例完成（{:.3f}s）...".format(time_gen - time_solute))
    #计算时间
    print(
        "总计用时：{:.3f}s\n测试用例个数：{}，正测试用例:{}个，负测试用例:{}个，正测试用例比例:{:.3f}%\n平均速度:{:.3f}个/s".format(time_gen - time_clear,
                                                                                             count, pos_count,
                                                                                             neg_count,
                                                                                             pos_count / count * 100,
                                                                                             count / (
                                                                                                         time_gen - time_clear)))
