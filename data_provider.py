import configuration

conf = configuration.config()


def preprocess_image():
    path = conf.data_path
    images = path + "images.txt"

    img_list = []
    with open(images) as fin:
        for eachline in fin.readlines():
            path = eachline.rstrip().split()[1]
            img_list.append(img_list)

    train_test_spits = path + "train_test_split.txt"
    train_list = []
    test_list = []
    with open(train_test_spits) as fin:
        for eachline in fin.readlines():
            elem = eachline.rstrip().split()
            if elem[1] == "1":
                train_list.append(img_list[int(elem[0])])
            else:
                test_list.append(img_list[int(elem[0])])
