import os
import shutil


class PrepareData:
    def __init__(self):
        pass

    def load_data_list(self, path):
        data_list, temp = [], ''
        with open(path, 'r') as f:
            while True:
                data = f.readline()
                if not data:
                    break
                data = data[:-1]
                data_list.append(data.split('/'))
        return data_list

    def move_data(self, path_data, c):
        if not(os.path.isdir('./dataset/' + c + path_data[0])):
            os.makedirs('./dataset/' + c + path_data[0])
        total_path = path_data[0] + '/' + path_data[1]
        shutil.move(src='./speech_commands_v0.01/' + total_path, dst='./dataset/' + c + path_data[0])


if __name__ == '__main__':
    p_data = PrepareData()
    test_list = p_data.load_data_list(path='./speech_commands_v0.01/testing_list.txt')
    print(test_list)
    print('test_list loaded')
    for t in test_list:
        p_data.move_data(path_data=t, c='test/')
        print(t, 'moved')

    valid_list = p_data.load_data_list(path='./speech_commands_v0.01/validation_list.txt')
    print('valid_list loaded')
    temp, class_list = '', []
    for v in valid_list:
        if v[0] != temp and class_list.count(v[0]) == 0:
            temp = v[0]
            class_list.append(temp)
        p_data.move_data(path_data=v, c='valid/')
        print(v, 'moved')

    print(class_list)
    if not(os.path.isdir('./dataset/train')):
        os.makedirs('./dataset/train')
    for c in class_list:
        shutil.move(src='./speech_commands_v0.01/'+c, dst='./dataset/train/')
        print(c, 'moved')
