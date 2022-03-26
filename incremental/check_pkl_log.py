import pickle


if __name__ == '__main__':
    with open('./cifar10_log/cifar10.pkl', 'rb') as f:
        a = pickle.load(f)
        for i in a:
            print(i)