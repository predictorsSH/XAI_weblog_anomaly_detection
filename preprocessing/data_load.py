import json

def json_load(file_path):
    print('file open!')

    data = []
    with open(file_path,'r') as f:
        for line in f:
            data.append(json.loads(line))

    return data

# def csv_load():


if __name__ == '__main__':

    file_path = '../data/si/SI_train.done'
    data = json_load(file_path)

