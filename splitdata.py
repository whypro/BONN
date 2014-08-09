def split_data(filename, classification_filename, features_filename):
    f = open(filename, 'r')
    cf = open(classification_filename, 'w')
    ff = open(features_filename, 'w')
    for line in f:
        classification = line.strip().split(' ')[0]
        features = line.strip().split(' ')[1:]
        cf.write(classification)
        cf.write('\n')
        ff.write(' '.join(features))
        ff.write('\n')
        
    ff.close()
    cf.close()
    f.close()
    

if __name__ == '__main__':
    split_data('data', 'data_c', 'data_f')
    split_data('data.test', 'data_c.test', 'data_f.test')
    

