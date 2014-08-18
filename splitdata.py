def split_data(filename, classification_filename, features_filename):
    f = open(filename, 'r')
    cf = open(classification_filename, 'w')
    ff = open(features_filename, 'w')
    for line in f:
        classification = line.strip().split(' ')[0]
        features = [feature.split(':')[1] for feature in line.strip().split(' ')[1:]]
        cf.write(classification)
        cf.write('\n')
        ff.write(' '.join(features))
        ff.write('\n')

    ff.close()
    cf.close()
    f.close()


if __name__ == '__main__':
    split_data('data', 'data_y', 'data_x')
    split_data('data.test', 'data_y.test', 'data_x.test')
    split_data('data.bin', 'data_y.bin', 'data_x.bin')
    split_data('data.bin.test', 'data_y.bin.test', 'data_x.bin.test')


