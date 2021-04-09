'''
Created on April , 2021
@author:
'''
import logging
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot
from sklearn import preprocessing
# from sklearn.decomposition import PCA
# from pyts.approximation import SymbolicFourierApproximation


def gen_sequence(id_df, seq_length, seq_cols):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """
    # for one id I put all the rows in a single matrix
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    # Iterate over two lists in parallel.

    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]

def gen_labels(id_df, seq_length, label):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """
    # For one id I put all the labels in a single matrix.
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    # I have to remove the first seq_length labels
    # because for one id the first sequence of seq_length size have as target
    # the last label (the previus ones are discarded).
    # All the next id's sequences will have associated step by step one label as target.
    return data_matrix[seq_length:num_elements, :]



class input_gen(object):
    '''
    class for data preparation (rps generator)
    '''

    def __init__(self, data_path_list, sequence_length, sensor_drop, piecewise_lin_ref=125, preproc=False, visualize=False):
        '''
        :param data_path_list: python list of four sub-dataset
        :param sequence_length: legnth of sequence (sliced time series)
        :param sensor_drop: sensors not to be considered
        :param piecewise_lin_ref: max rul value (if real rul value is larger than piecewise_lin_ref,
        then the rul value is piecewise_lin_ref)
        :param preproc: preprocessing
        '''
        # self.__logger = logging.getLogger('data preparation for using it as the network input')
        self.data_path_list = data_path_list
        self.sequence_length = sequence_length
        self.sensor_drop = sensor_drop
        self.preproc = preproc
        self.piecewise_lin_ref = piecewise_lin_ref
        self.visualize = visualize


        ## Assign columns name
        cols = ['unit_nr', 'cycles', 'os_1', 'os_2', 'os_3']
        cols += ['sensor_{0:02d}'.format(s + 1) for s in range(26)]
        col_rul = ['RUL_truth']

        train_FD = pd.read_csv(self.data_path_list[0], sep=' ', header=None,
                               names=cols, index_col=False)
        test_FD = pd.read_csv(self.data_path_list[1], sep=' ', header=None,
                              names=cols, index_col=False)
        RUL_FD = pd.read_csv(self.data_path_list[2], sep=' ', header=None,
                             names=col_rul, index_col=False)

        ## Calculate RUL and append to train data
        # get the time of the last available measurement for each unit
        mapper = {}
        for unit_nr in train_FD['unit_nr'].unique():
            mapper[unit_nr] = train_FD['cycles'].loc[train_FD['unit_nr'] == unit_nr].max()

        # calculate RUL = time.max() - time_now for each unit
        train_FD['RUL'] = train_FD['unit_nr'].apply(lambda nr: mapper[nr]) - train_FD['cycles']
        # piecewise linear for RUL labels
        train_FD['RUL'].loc[(train_FD['RUL'] > self.piecewise_lin_ref)] = self.piecewise_lin_ref

        # Cut max RUL ground truth
        RUL_FD['RUL_truth'].loc[(RUL_FD['RUL_truth'] > self.piecewise_lin_ref)] = self.piecewise_lin_ref

        ## Excluse columns which only have NaN as value
        # nan_cols = ['sensor_{0:02d}'.format(s + 22) for s in range(5)]
        cols_nan = train_FD.columns[train_FD.isna().any()].tolist()
        # print('Columns with all nan: \n' + str(cols_nan) + '\n')
        cols_const = [col for col in train_FD.columns if len(train_FD[col].unique()) <= 2]
        # print('Columns with all const values: \n' + str(cols_const) + '\n')

        ## Drop exclusive columns
        # train_FD = train_FD.drop(columns=cols_const + cols_nan)
        # test_FD = test_FD.drop(columns=cols_const + cols_nan)

        train_FD = train_FD.drop(columns=cols_const + cols_nan + sensor_drop)

        test_FD = test_FD.drop(columns=cols_const + cols_nan + sensor_drop)


        if self.preproc == True:
            ## preprocessing(normailization for the neural networks)
            min_max_scaler = preprocessing.MinMaxScaler()
            # for the training set
            # train_FD['cycles_norm'] = train_FD['cycles']
            cols_normalize = train_FD.columns.difference(['unit_nr', 'cycles', 'os_1', 'os_2', 'RUL'])

            norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_FD[cols_normalize]),
                                         columns=cols_normalize,
                                         index=train_FD.index)
            join_df = train_FD[train_FD.columns.difference(cols_normalize)].join(norm_train_df)
            train_FD = join_df.reindex(columns=train_FD.columns)

            # for the test set
            # test_FD['cycles_norm'] = test_FD['cycles']
            cols_normalize_test = test_FD.columns.difference(['unit_nr', 'cycles', 'os_1', 'os_2'])
            # print ("cols_normalize_test", cols_normalize_test)
            norm_test_df = pd.DataFrame(min_max_scaler.transform(test_FD[cols_normalize_test]), columns=cols_normalize_test,
                                        index=test_FD.index)
            test_join_df = test_FD[test_FD.columns.difference(cols_normalize_test)].join(norm_test_df)
            test_FD = test_join_df.reindex(columns=test_FD.columns)
            test_FD = test_FD.reset_index(drop=True)
        else:
            # print ("No preprocessing")
            pass

        # Specify the columns to be used
        sequence_cols_train = train_FD.columns.difference(['unit_nr', 'cycles', 'os_1', 'os_2', 'RUL'])
        sequence_cols_test = test_FD.columns.difference(['unit_nr', 'os_1', 'os_2', 'cycles'])



        ## generator for the sequences
        # transform each id of the train dataset in a sequence
        seq_gen = (list(gen_sequence(train_FD[train_FD['unit_nr'] == id], self.sequence_length, sequence_cols_train))
                   for id in train_FD['unit_nr'].unique())

        # generate sequences and convert to numpy array in training set
        seq_array_train = np.concatenate(list(seq_gen)).astype(np.float32)
        self.seq_array_train = seq_array_train.transpose(0, 2, 1) # shape = (samples, sensors, sequences)
        # print("seq_array_train.shape", self.seq_array_train.shape)

        # generate label of training samples
        label_gen = [gen_labels(train_FD[train_FD['unit_nr'] == id], self.sequence_length, ['RUL'])
                     for id in train_FD['unit_nr'].unique()]
        self.label_array_train = np.concatenate(label_gen).astype(np.float32)

        # generate sequences and convert to numpy array in test set (only the last sequence for each engine in test set)
        seq_array_test_last = [test_FD[test_FD['unit_nr'] == id][sequence_cols_test].values[-self.sequence_length:]
                               for id in test_FD['unit_nr'].unique() if
                               len(test_FD[test_FD['unit_nr'] == id]) >= self.sequence_length]

        seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)
        self.seq_array_test_last = seq_array_test_last.transpose(0, 2, 1) # shape = (samples, sensors, sequences)
        # print("seq_array_test_last.shape", self.seq_array_test_last.shape)

        # generate label of test samples
        y_mask = [len(test_FD[test_FD['unit_nr'] == id]) >= self.sequence_length for id in test_FD['unit_nr'].unique()]
        label_array_test_last = RUL_FD['RUL_truth'][y_mask].values
        self.label_array_test = label_array_test_last.reshape(label_array_test_last.shape[0], 1).astype(np.float32)


        ## Visualize Run-2-failure TS of the first engine in the training set.(Please deactivate after understanding)
        if self.visualize == True:
            # R2F TS of the first engine
            pd.DataFrame(train_FD[train_FD['unit_nr'] == 1][sequence_cols_train].values,
                             columns=sequence_cols_train).plot(subplots=True, figsize=(15, 15))

            # The last sequences sliced from each TS (of the first engine)
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']
            colors = colors + colors + colors

            seq_gen = (
            list(gen_sequence(train_FD[train_FD['unit_nr'] == id], self.sequence_length, sequence_cols_train))
            for id in train_FD['unit_nr'].unique())

            seq_list_engine = list(seq_gen)
            seq_engine_1_array = np.asarray(seq_list_engine[0])

            last_seq_engine_1_array = seq_engine_1_array[-1, :, :]
            fig_ts = plt.figure(figsize=(15, 15))
            for s in range(last_seq_engine_1_array.shape[1]):
                seq_s = last_seq_engine_1_array[:, s]
                # plt.subplot(last_seq_engine_1_array.shape[1],(s//4) + 1, (s%4)+1)
                plt.subplot(4, 4, s + 1)
                plt.plot(seq_s, "y", label=sequence_cols_train[s], color=colors[s])
                plt.legend()

            plt.xlabel("time(cycles)")
            plt.show()


    def rps(self, thres_type=None, thres_percentage=50, flatten=False, visualize=True):
        '''
        generate RPs from sequences
        :param thres_type:  ‘point’, ‘distance’ or None (default = None)
        :param thres_percentage:
        :param flatten:
        :param visualize: visualize generated RPs (first training sample)
        :return: PRs (samples for NNs and their label)
        '''

        # Recurrence plot transformation for training samples
        rp_train = RecurrencePlot(threshold=thres_type, percentage=thres_percentage,
                                  flatten=flatten)

        rp_list = []
        for idx in range(self.seq_array_train.shape[0]):
            temp_mts = self.seq_array_train[idx]
            # print (temp_mts.shape)
            X_rp_temp = rp_train.fit_transform(temp_mts)
            # print (X_rp_temp.shape)
            rp_list.append(X_rp_temp)

        rp_train_samples = np.stack(rp_list, axis=0)

        # Recurrence plot transformation for test samples
        rp_test = RecurrencePlot(threshold=thres_type, percentage=thres_percentage, flatten=flatten)
        rp_list = []
        for idx in range(self.seq_array_test_last.shape[0]):
            temp_mts = self.seq_array_test_last[idx]
            # print (temp_mts.shape)
            X_rp_temp = rp_test.fit_transform(temp_mts)
            # print (X_rp_temp.shape)
            rp_list.append(X_rp_temp)
        rp_test_samples = np.stack(rp_list, axis=0)

        label_array_train = self.label_array_train
        label_array_test = self.label_array_test

        # Visualize RPs of the last sequences sliced from each TS (of the first engine)
        if visualize == True:
            X_rp = rp_train_samples[-1]
            plt.figure(figsize=(15, 15))
            for s in range(len(X_rp)):
                # plt.subplot(last_seq_engine_1_array.shape[1],(s//4) + 1, (s%4)+1)
                plt.subplot(4, 4, s + 1)
                if flatten == True:
                    img = np.atleast_2d(X_rp[s])
                    plt.imshow(img, extent=(0, img.shape[1], 0, round(img.shape[1]/9)))
                else:
                    plt.imshow(X_rp[s], origin='lower')
                # plt.legend()
            plt.show()

        return  rp_train_samples, label_array_train, rp_test_samples, label_array_test









