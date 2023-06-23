import numpy as np
import random
import matplotlib.pyplot as plt
import itertools
import tkinter
import matplotlib as mpl

mpl.use('TkAgg')


class Plots:
    def __init__(self, n, z, alg):
        self.z = z
        self.n = n
        self.alg = alg  # 'rs' or 'sha1' or 'sha2' or 'nc'


    def convert_n_to_k(self, n_max):
        return np.array([np.power(2, i + 1) for i in range(n_max)])


    def convert_z_to_q(self, z):
        return np.array([np.power(2, i) for i in z])

    def convert_z_correct(self, z_max):
        z_r = []
        for i in range(z_max):
            if(i == 0):
                z_r.append(8)
            elif(i == 1):
                z_r.append(16)
            elif(i == 2):
                z_r.append(32)
                
        return z_r 

    def plot_prob(self, prob_data_2D, plot_what, plot_how_many = 'all'):
        alg = self.alg
        n = self.n
        z = self.z
        labels_k = self.convert_n_to_k(n)
        z_r = self.convert_z_correct(z)
        labels_q = self.convert_z_to_q(z_r)
        title = ''
        save_z = ''  # file name
        save_n = ''
        save_q = ''
        save_k = ''
        label = ''
        fontdict = {'fontsize': 9}  # font size of titles
        y_label = 'FP-Probability'

        if alg == 'rs':
            title = 'FP-Probability of Reed Solomon'
            save_z = 'RS_Prob_SameZ_DiffN.png'
            save_n = 'RS_Prob_SameN_DiffZ.png'
            save_q = 'RS_Prob_SameQ_DiffK.png'
            save_k = 'RS_Prob_SameK_DiffQ.png'
            label = 'Calculated FP-Probability RS'
            y_calc = 0.004 #@Ed richtigen wert bitte einfügen
        elif alg == 'sha1':
            title = 'FP-Probability of Sha1'
            save_z = 'SHA1_Prob_SameZ_DiffN.png'
            save_n = 'SHA1_Prob_SameN_DiffZ.png'
            save_q = 'SHA1_Prob_SameQ_DiffK.png'
            save_k = 'SHA1_Prob_SameK_DiffQ.png'
            label = 'Calculated FP-Probability SHA1'
            y_calc = 0.004 #@Ed richtigen wert bitte einfügen
        elif alg == 'sha2':
            title = 'FP-Probability of Sha2'
            save_z = 'SHA2_Prob_SameZ_DiffN.png'
            save_n = 'SHA2_Prob_SameN_DiffZ.png'
            save_q = 'SHA2_Prob_SameQ_DiffK.png'
            save_k = 'SHA2_Prob_SameK_DiffQ.png'
            label = 'Calculated FP-Probability SHA2'
            y_calc = 0.004 #@Ed richtigen wert bitte einfügen
        elif alg == 'nc':
            title = 'FP-Probability of No Code'
            save_z = 'NC_Prob_SameZ_DiffN.png'
            save_n = 'NC_Prob_SameN_DiffZ.png'
            save_q = 'NC_Prob_SameQ_DiffK.png'
            save_k = 'NC_Prob_SameK_DiffQ.png'
            label = 'Calculated FP-Probability NC'
            y_calc = 0.004 #@Ed richtigen wert bitte einfügen

        num_columns = prob_data_2D.shape[1]
        num_rows = prob_data_2D.shape[0]

        #x_r = np.arange(num_rows) + 1
        x_r = np.array(self.convert_z_correct(z))
        x_c = np.arange(num_columns) + 1

        if plot_what == 'z':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            bar_width = 0.8 / num_rows
            if plot_how_many == 'all':
                for i in range(num_rows):
                    bar_position = x_c + i * bar_width
                    ax.bar(bar_position, prob_data_2D[i, :], width=bar_width, label='z = {}'.format(x_r[i]))
                ax.set_title('FP-Probability for different symbol sizes z', fontdict=fontdict)
            else:  # plot z for a single n
                bar_position = x_c + (num_rows - 1) * bar_width / 2
                ax.bar(bar_position, prob_data_2D[plot_how_many, :], width=bar_width, label='z = {}'.format(x_r[plot_how_many]))
                ax.set_title('FP-Probability for symbol size z = {}'.format(x_r[plot_how_many]), fontdict=fontdict)

            ax.axhline(y=y_calc, c='r', ls=':', label=label)  # dotted line = calculated FP-Average
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('n with k = 2^n', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticks(x_c + (num_rows - 1) * bar_width / 2)
            ax.set_xticklabels(x_c)

            # plt.savefig(save_z)
            # np.save(save_z, prob_data_2D)
            plt.show()

        elif plot_what == 'q':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            bar_width = 0.8 / num_rows
            if plot_how_many == 'all':
                for i in range(num_rows):
                    bar_position = x_c + i * bar_width
                    ax.bar(bar_position, prob_data_2D[i, :], width=bar_width, label='q = 2^{}'.format(x_r[i]))
                ax.set_title('FP-Probability for different q-ary bases', fontdict=fontdict)
            else:  # plot q for a single k
                bar_position = x_c + (num_rows - 1) * bar_width / 2
                ax.bar(bar_position, prob_data_2D[plot_how_many, :], width=bar_width,
                       label='q = 2^{}'.format(x_r[plot_how_many]))
                ax.set_title('FP-Probability for q-ary basis q = 2^{}'.format(x_r[plot_how_many]), fontdict=fontdict)

            ax.axhline(y=y_calc, c='r', ls=':', label=label)
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('Code word length k', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticks(x_c + (num_rows - 1) * bar_width / 2)
            ax.set_xticklabels(['$2^{{{}}}$'.format(b) for b in x_c])  # = labels_k

            # plt.savefig(save_q)
            # np.save(save_q, prob_data_2D)
            plt.show()

        elif plot_what == 'n':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            bar_width = 0.8 / num_columns
            if plot_how_many == 'all':
                for j in range(num_columns):
                    bar_position = x_r + j * bar_width
                    ax.bar(bar_position, prob_data_2D[:, j], width=bar_width, label='n = {}'.format(j+1))
                ax.set_title('FP-Probability for different code lengths n', fontdict=fontdict)
            else:
                bar_position = x_r + (num_columns - 1) * bar_width / 2  # equals xticks
                ax.bar(bar_position, prob_data_2D[:, plot_how_many], width=6 * bar_width, label='n = {}'.format(plot_how_many))
                ax.set_title('FP-Probability for code length n = {}'.format(plot_how_many), fontdict=fontdict)

            ax.axhline(y=y_calc, c='r', ls=':', label=label)
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('Symbol size z', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticks(x_r + (num_columns - 1) * bar_width / 2)
            ax.set_xticklabels(x_r)

            # plt.savefig(save_n)
            # np.save(save_n, prob_data_2D)

            plt.show()

        elif plot_what == 'k':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            bar_width = 0.8 / num_columns
            if plot_how_many == 'all':
                for j in range(num_columns):
                    bar_position = x_r + j * bar_width
                    ax.bar(bar_position, prob_data_2D[:, j], width=bar_width, label='k = 2^{}'.format(j + 1))
                ax.set_title('FP-Probability for different code word lengths k', fontdict=fontdict)
            else:
                bar_position = x_r + (num_columns - 1) * bar_width / 2  # equals xticks
                ax.bar(bar_position, prob_data_2D[:, plot_how_many], width=6 * bar_width, label='k = 2^{}'.format(plot_how_many + 1))
                ax.set_title('FP-Probability for code word lengths k = 2^{}'.format(plot_how_many + 1), fontdict=fontdict)

            ax.axhline(y=y_calc, c='r', ls=':', label=label)
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('q-ary basis q', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticks(x_r + (num_columns - 1) * bar_width / 2)
            ax.set_xticklabels(['$2^{{{}}}$'.format(b) for b in x_r])

            # plt.savefig(save_k)
            # np.save(save_k, prob_data_2D)

            plt.show()

    def plot_prob_time(self, prob_data_2D, plot_what, plot_how_many = 'all'):
        alg = self.alg
        n = self.n
        z = self.z
        labels_k = self.convert_n_to_k(n)
        z_r = self.convert_z_correct(z)
        labels_q = self.convert_z_to_q(z_r)

        title = ''
        save_z = ''
        save_n = ''
        save_q = ''
        save_k = ''
        fontdict = {'fontsize': 9}  # font size of titles
        y_label = 'Time in s'

        if alg == 'rs':
            title = 'Calculation-Time of Reed Solomon'
            save_n = 'RS_Time_SameZ_DiffN.png'
            save_z = 'RS_Time_SameN_DiffZ.png'
            save_q = 'RS_Time_SameQ_DiffK.png'
            save_k = 'RS_Time_SameK_DiffQ.png'
        elif alg == 'sha1':
            title = 'Calculation-Time of SHA1'
            save_z = 'SHA1_Time_SameZ_DiffN.png'
            save_n = 'SHA1_Time_SameN_DiffZ.png'
            save_q = 'SHA1_Time_SameQ_DiffK.png'
            save_k = 'SHA1_Time_SameK_DiffQ.png'
        elif alg == 'sha2':
            title = 'Calculation-Time of SHA2'
            save_z = 'SHA2_Time_SameZ_DiffN.png'
            save_n = 'SHA2_Time_SameN_DiffZ.png'
            save_q = 'SHA2_Time_SameQ_DiffK.png'
            save_k = 'SHA2_Time_SameK_DiffQ.png'
        elif alg == 'nc':
            title = 'Calculation-Time of No Code'
            save_z = 'NC_Time_SameZ_DiffN.png'
            save_n = 'NC_Time_SameN_DiffZ.png'
            save_q = 'NC_Time_SameQ_DiffK.png'
            save_k = 'NC_Time_SameK_DiffQ.png'

        num_rows = prob_data_2D.shape[0]
        num_columns = prob_data_2D.shape[1]

        #x_r = np.arange(num_rows) + 1
        x_r = np.array(self.convert_z_correct(z))
        x_c = np.arange(num_columns) + 1
        x_cpos = np.zeros(num_columns)
        for j in range(num_columns):
            x_cpos[j] = np.power(2, j+1)

        if plot_what == 'z':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            bar_width = 0.8 / num_rows
            if plot_how_many == 'all':
                for i in range(num_rows):
                    bar_position = x_c + i * bar_width
                    ax.bar(bar_position, prob_data_2D[i, :], width=0.9*bar_width, label='z = {}'.format(x_r[i]))
                ax.set_title('Calculation Time for different symbol sizes z', fontdict=fontdict)
            else:  # plot z for a single n
                bar_position = x_c + (num_rows - 1) * bar_width / 2
                ax.bar(bar_position, prob_data_2D[plot_how_many, :], width=bar_width, label='z = {}'.format(x_r[plot_how_many]))
                ax.set_title('Calculation Time for symbol size z = {}'.format(x_r[plot_how_many]), fontdict=fontdict)

            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('n with k = 2^n', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticks(x_c + (num_rows - 1) * bar_width / 2)
            ax.set_xticklabels(x_c)

            #plt.savefig(save_z)
            # np.save(save_z, prob_data_2D)
            plt.show()

        elif plot_what == 'q':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            bar_width = 0.8 / num_rows
            if plot_how_many == 'all':
                for i in range(num_rows):
                    bar_position = x_cpos + i * bar_width
                    ax.bar(bar_position, prob_data_2D[i, :], width=0.9*bar_width, label='q = 2^{}'.format(x_r[i]))
                ax.set_title('Calculation Time for different q-ary bases', fontdict=fontdict)
            else:  # plot q for a single k
                bar_position = x_c + (num_rows - 1) * bar_width / 2
                ax.bar(bar_position, prob_data_2D[plot_how_many, :], width=bar_width,
                       label='q = 2^{}'.format(x_r[plot_how_many]))
                ax.set_title('Calculation Time for q-ary basis q = 2^{}'.format(x_r[plot_how_many]), fontdict=fontdict)

            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('Code word length k', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticks(x_cpos + (num_rows - 1) * bar_width / 2)
            ax.set_xticklabels(['$2^{{{}}}$'.format(b) for b in x_c])  # = labels_k

            #plt.savefig(save_q)
            # np.save(save_q, prob_data_2D)
            plt.show()

        elif plot_what == 'n':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            bar_width = 0.8 / num_columns
            if plot_how_many == 'all':
                for j in range(num_columns):
                    bar_position = x_r + j * bar_width
                    ax.bar(bar_position, prob_data_2D[:, j], width=0.9*bar_width, label='n = {}'.format(j+1))
                ax.set_title('Calculation Time for different code lengths n', fontdict=fontdict)
            else:
                bar_position = x_r + (num_columns - 1) * bar_width / 2  # equals xticks
                ax.bar(bar_position, prob_data_2D[:, plot_how_many], width=6 * bar_width, label='n = {}'.format(plot_how_many))
                ax.set_title('Calculation Time for code length n = {}'.format(plot_how_many), fontdict=fontdict)

            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('Symbol size z', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticks(x_r + (num_columns - 1) * bar_width / 2)
            ax.set_xticklabels(x_r)

            #plt.savefig(save_n)
            # np.save(save_n, prob_data_2D)
            plt.show()

        elif plot_what == 'k':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            bar_width = 0.8 / num_columns
            if plot_how_many == 'all':
                for j in range(num_columns):
                    bar_position = x_r + j * bar_width
                    ax.bar(bar_position, prob_data_2D[:, j], width=0.9*bar_width, label='k = 2{}'.format(j+1))
                ax.set_title('Calculation Time for different code word lengths k', fontdict=fontdict)
            else:
                bar_position = x_r + (num_columns - 1) * bar_width / 2  # equals xticks
                ax.bar(bar_position, prob_data_2D[:, plot_how_many], width=6 * bar_width, label='k = 2^{}'.format(plot_how_many + 1))
                ax.set_title('Calculation Time for code word lengths k = 2^{}'.format(plot_how_many + 1), fontdict=fontdict)

            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('q-ary basis q', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticks(x_r + (num_columns - 1) * bar_width / 2)
            ax.set_xticklabels(['$2^{{{}}}$'.format(b) for b in x_r])

            # plt.savefig(save_k)
            # np.save(save_k, prob_data_2D)
            plt.show()

    def plot_prob_thrp(self, prob_data_2D, plot_what, plot_how_many = 'all'):
        alg = self.alg
        n = self.n
        z = self.z
        labels_k = self.convert_n_to_k(n)
        z_r = self.convert_z_correct(z)
        labels_q = self.convert_z_to_q(z_r)

        title = ''
        save_z = ''
        save_n = ''
        save_q = ''
        save_k = ''
        fontdict = {'fontsize': 9}  # font size of titles
        y_label = 'Thrp in GB/s'

        if alg == 'rs':
            title = 'Calculation-Throughput of Reed Solomon'
            save_n = 'RS_Thrp_SameZ_DiffN.png'
            save_z = 'RS_Thrp_SameN_DiffZ.png'
            save_q = 'RS_Thrp_SameQ_DiffK.png'
            save_k = 'RS_Thrp_SameK_DiffQ.png'
        elif alg == 'sha1':
            title = 'Calculation-Throughput of SHA1'
            save_z = 'SHA1_Thrp_SameZ_DiffN.png'
            save_n = 'SHA1_Thrp_SameN_DiffZ.png'
            save_q = 'SHA1_Thrp_SameQ_DiffK.png'
            save_k = 'SHA1_Thrp_SameK_DiffQ.png'
        elif alg == 'sha2':
            title = 'Calculation-Throughput of SHA2'
            save_z = 'SHA2_Thrp_SameZ_DiffN.png'
            save_n = 'SHA2_Thrp_SameN_DiffZ.png'
            save_q = 'SHA2_Thrp_SameQ_DiffK.png'
            save_k = 'SHA2_Thrp_SameK_DiffQ.png'
        elif alg == 'nc':
            title = 'Calculation-Throughput of No Code'
            save_z = 'NC_Thrp_SameZ_DiffN.png'
            save_n = 'NC_Thrp_SameN_DiffZ.png'
            save_q = 'NC_Thrp_SameQ_DiffK.png'
            save_k = 'NC_Thrp_SameK_DiffQ.png'

        num_rows = prob_data_2D.shape[0]
        num_columns = prob_data_2D.shape[1]

        #x_r = np.arange(num_rows) + 1
        x_r = np.array(self.convert_z_correct(z))
        x_c = np.arange(num_columns) + 1
        x_cpos = np.zeros(num_columns)
        for j in range(num_columns):
            x_cpos[j] = np.power(2, j+1)

        if plot_what == 'z':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            bar_width = 0.8 / num_rows
            if plot_how_many == 'all':
                for i in range(num_rows):
                    bar_position = x_c + i * bar_width
                    ax.bar(bar_position, prob_data_2D[i, :], width=0.9*bar_width, label='z = {}'.format(x_r[i]))
                ax.set_title('Calculation Thrpt for different symbol sizes z', fontdict=fontdict)
            else:  # plot z for a single n
                bar_position = x_c + (num_rows - 1) * bar_width / 2
                ax.bar(bar_position, prob_data_2D[plot_how_many, :], width=bar_width, label='z = {}'.format(x_r[plot_how_many]))
                ax.set_title('Calculation Thrpt for symbol size z = {}'.format(x_r[plot_how_many]), fontdict=fontdict)

            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('n with k = 2^n', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticks(x_c + (num_rows - 1) * bar_width / 2)
            ax.set_xticklabels(x_c)

            #plt.savefig(save_z)
            # np.save(save_z, prob_data_2D)
            plt.show()

        elif plot_what == 'q':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            bar_width = 0.8 / num_rows
            if plot_how_many == 'all':
                for i in range(num_rows):
                    bar_position = x_cpos + i * bar_width
                    ax.bar(bar_position, prob_data_2D[i, :], width=0.9*bar_width, label='q = 2^{}'.format(x_r[i]))
                ax.set_title('Calculation Thrpt for different q-ary bases', fontdict=fontdict)
            else:  # plot q for a single k
                bar_position = x_c + (num_rows - 1) * bar_width / 2
                ax.bar(bar_position, prob_data_2D[plot_how_many, :], width=bar_width,
                       label='q = 2^{}'.format(x_r[plot_how_many]))
                ax.set_title('Calculation Thrpt for q-ary basis q = 2^{}'.format(x_r[plot_how_many]), fontdict=fontdict)

            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('Code word length k', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticks(x_cpos + (num_rows - 1) * bar_width / 2)
            ax.set_xticklabels(['$2^{{{}}}$'.format(b) for b in x_c]) # = labels_k

            #plt.savefig(save_q)
            # np.save(save_q, prob_data_2D)
            plt.show()

        elif plot_what == 'n':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            bar_width = 0.8 / num_columns
            if plot_how_many == 'all':
                for j in range(num_columns):
                    bar_position = x_r + j * bar_width
                    ax.bar(bar_position, prob_data_2D[:, j], width=0.9*bar_width, label='n = {}'.format(j+1))
                ax.set_title('Calculation Thrpt for different code lengths n', fontdict=fontdict)
            else:
                bar_position = x_r + (num_columns - 1) * bar_width / 2  # equals xticks
                ax.bar(bar_position, prob_data_2D[:, plot_how_many], width=6 * bar_width, label='n = {}'.format(plot_how_many))
                ax.set_title('Calculation Thrpt for code length n = {}'.format(plot_how_many), fontdict=fontdict)

            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('Symbol size z', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticks(x_r + (num_columns - 1) * bar_width / 2)
            ax.set_xticklabels(x_r)

            #plt.savefig(save_n)
            # np.save(save_n, prob_data_2D)
            plt.show()

        elif plot_what == 'k':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            bar_width = 0.8 / num_columns
            if plot_how_many == 'all':
                for j in range(num_columns):
                    bar_position = x_r + j * bar_width
                    ax.bar(bar_position, prob_data_2D[:, j], width=0.9*bar_width, label='k = 2^{}'.format(j + 1))
                ax.set_title('Calculation Thrpt for different code word lengths k', fontdict=fontdict)
            else:
                bar_position = x_r + (num_columns - 1) * bar_width / 2  # equals xticks
                ax.bar(bar_position, prob_data_2D[:, plot_how_many], width=6 * bar_width, label='k = 2^{}'.format(plot_how_many + 1))
                ax.set_title('Calculation Thrpt for code word lengths k = 2^{}'.format(plot_how_many + 1), fontdict=fontdict)

            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('q-ary bases q', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticks(x_r + (num_columns - 1) * bar_width / 2)
            ax.set_xticklabels(['$2^{{{}}}$'.format(b) for b in x_r])

            # plt.savefig(save_k)
            # np.save(save_k, prob_data_2D)
            plt.show()

    def plot_prob_pt(self, prob_data_2D, plot_what):
        alg = self.alg
        n = self.n
        z = self.z
        labels_k = self.convert_n_to_k(n)
        z_r = self.convert_z_correct(z)
        labels_q = self.convert_z_to_q(z_r)
        title = ''
        save_z = ''
        save_n = ''
        save_q = ''
        save_k = ''
        fontdict = {'fontsize': 9}  # font size of titles
        y_label = 'FP-Probability\n*CalcTime in s'

        if alg == 'rs':
            title = 'FP-Probability * Calculation Time of Reed Solomon'
            save_n = 'RS_pt_SameZ_DiffN.png'
            save_z = 'RS_pt_SameN_DiffZ.png'
            save_q = 'RS_pt_SameQ_DiffK.png'
            save_k = 'RS_pt_SameK_DiffQ.png'
        elif alg == 'sha1':
            title = 'FP-Probability * Calculation Time of SHA1'
            save_n = 'SHA1_pt_SameZ_DiffN.png'
            save_z = 'SHA1_pt_SameN_DiffZ.png'
            save_q = 'SHA1_pt_SameQ_DiffK.png'
            save_k = 'SHA1_pt_SameK_DiffQ.png'
        elif alg == 'sha2':
            title = 'FP-Probability * Calculation Time of SHA2'
            save_n = 'SHA2_pt_SameZ_DiffN.png'
            save_z = 'SHA2_pt_SameN_DiffZ.png'
            save_q = 'SHA2_pt_SameQ_DiffK.png'
            save_k = 'SHA2_pt_SameK_DiffQ.png'
        elif alg == 'nc':
            title = 'FP-Probability * Calculation Time of No Code'
            save_n = 'NC_pt_SameZ_DiffN.png'
            save_z = 'NC_pt_SameN_DiffZ.png'
            save_q = 'NC_pt_SameQ_DiffK.png'
            save_k = 'NC_pt_SameK_DiffQ.png'

        num_rows = prob_data_2D.shape[0]
        num_columns = prob_data_2D.shape[1]

        #x_r = np.arange(num_rows) + 1
        x_r = np.array(self.convert_z_correct(z))
        x_c = np.arange(num_columns) + 1

        if plot_what == 'z':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            bar_width = 0.8 / num_rows

            for i in range(num_rows):
                bar_position = x_c + i * bar_width
                ax.bar(bar_position, prob_data_2D[i, :], width=0.9*bar_width, label='z = {}'.format(i + 1))

            ax.set_title('FP-Probability * Calculation Time for different symbol sizes z', fontdict=fontdict)
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('Code length n', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticks(x_r + (num_columns - 1) * bar_width / 2)
            ax.set_xticklabels(x_r)

            # plt.savefig(save_z)
            # np.save(save_z, prob_data_2D)
            plt.show()

        elif plot_what == 'q':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            bar_width = 0.8 / num_rows

            for i in range(num_rows):
                bar_position = x_c + i * bar_width
                ax.bar(bar_position, prob_data_2D[i, :], width=0.9 * bar_width, label='q = {}'.format(labels_q[i]))

            ax.set_title('FP-Probability * Calculation Time for different q-ary bases', fontdict=fontdict)
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('Code length n', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticks(x_r + (num_columns - 1) * bar_width / 2)
            ax.set_xticklabels(labels_k)

            # plt.savefig(save_q)
            # np.save(save_q, prob_data_2D)
            plt.show()

        elif plot_what == 'n':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            bar_width = 0.8 / num_columns
            for j in range(num_columns):
                bar_position = x_r + j * bar_width
                ax.bar(bar_position, prob_data_2D[:, j], width=0.9*bar_width, label='n = {}'.format(j+1))

            ax.set_title('FP-Probability * Calculation Time for different code lengths n', fontdict=fontdict)
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('Symbol size n', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticks(x_r + (num_columns - 1) * bar_width / 2)
            ax.set_xticklabels(x_r)

            # plt.savefig(save_n)
            # np.save(save_n, prob_data_2D)

            plt.show()

        elif plot_what == 'k':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            bar_width = 0.8 / num_columns
            for j in range(num_columns):
                bar_position = x_r + j * bar_width
                ax.bar(bar_position, prob_data_2D[:, j], width=0.9*bar_width, label='k = {}'.format(labels_k[j]))

            ax.set_title('FP-Probability * Calculation Time for different code word lengths k', fontdict=fontdict)
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('Symbol size n', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticks(x_r + (num_columns - 1) * bar_width / 2)
            ax.set_xticklabels(labels_q)

            # plt.savefig(save_k)
            # np.save(save_k, prob_data_2D)

            plt.show()

    def plot_mean(self, prob_data_2D, plot_what):
        alg = self.alg
        n = self.n
        z = self.z
        labels_k = self.convert_n_to_k(n)
        z_r = self.convert_z_correct(z)
        labels_q = self.convert_z_to_q(z_r)
        title = ''
        save_z = ''  # file name
        save_n = ''
        save_q = ''
        save_k = ''
        label = ''
        fontdict = {'fontsize': 9}  # font size of titles
        y_label = 'Average FP-Probability'

        if alg == 'rs':
            title = 'Average FP-Probability of Reed Solomon'
            save_z = 'RS_Prob_SameZ_DiffN.png'
            save_n = 'RS_Prob_SameN_DiffZ.png'
            save_q = 'RS_Prob_SameQ_DiffK.png'
            save_k = 'RS_Prob_SameK_DiffQ.png'
            label = 'Calculated FP-Probability RS'
            y_calc = 0.004  # @Ed richtigen wert bitte einfügen
        elif alg == 'sha1':
            title = 'Average FP-Probability of Sha1'
            save_z = 'SHA1_Prob_SameZ_DiffN.png'
            save_n = 'SHA1_Prob_SameN_DiffZ.png'
            save_q = 'SHA1_Prob_SameQ_DiffK.png'
            save_k = 'SHA1_Prob_SameK_DiffQ.png'
            label = 'Calculated FP-Probability SHA1'
            y_calc = 0.004  # @Ed richtigen wert bitte einfügen
        elif alg == 'sha2':
            title = 'Average FP-Probability of Sha2'
            save_z = 'SHA2_Prob_SameZ_DiffN.png'
            save_n = 'SHA2_Prob_SameN_DiffZ.png'
            save_q = 'SHA2_Prob_SameQ_DiffK.png'
            save_k = 'SHA2_Prob_SameK_DiffQ.png'
            label = 'Calculated FP-Probability SHA2'
            y_calc = 0.004  # @Ed richtigen wert bitte einfügen
        elif alg == 'nc':
            title = 'Average FP-Probability of No Code'
            save_z = 'NC_Prob_SameZ_DiffN.png'
            save_n = 'NC_Prob_SameN_DiffZ.png'
            save_q = 'NC_Prob_SameQ_DiffK.png'
            save_k = 'NC_Prob_SameK_DiffQ.png'
            label = 'Calculated FP-Probability NC'
            y_calc = 0.004  # @Ed richtigen wert bitte einfügen

        num_rows = prob_data_2D.shape[0]
        num_columns = prob_data_2D.shape[1]

        mean_c = []
        mean_r = []

        for k in range(num_columns):
            sum = 0
            for l in range(num_rows):
                sum += prob_data_2D[l][k]
            mean = sum / num_columns
            mean_r.append(mean)

        for m in range(num_rows):
            sum = 0
            for n in range(num_columns):
                sum += prob_data_2D[m][n]
            mean = sum / num_rows
            mean_c.append(mean)

        x_r = np.arange(len(mean_r)) + 1
        x_c = np.arange(len(mean_c)) + 1

        # avg_r = np.array(mean_r)
        # avg_c = np.array(mean_c)
        # num_col_c = avg_c[0]
        # num_col_r = avg_r[0]
        # x_r = np.arange(num_col_r) + 1
        # x_c = np.arange(num_col_c) + 1

        if plot_what == 'n':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            bar_width = 0.8 / x_c
            ax.bar(x_r, mean_r, width=0.9*bar_width)
            ax.axhline(y=y_calc, c='r', ls=':', label=label)
            ax.grid()
            ax.set_title('Average FP-Probability of different code lengths n', fontdict=fontdict)
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('Code length n', fontsize=8)
            ax.legend()

            # plt.savefig(save_n)
            # np.save(save_n, prob_data_2D)
            plt.show()

        elif plot_what == 'k':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            bar_width = 0.8 / x_c
            ax.bar(x_r, mean_r, width=0.9*bar_width)
            ax.axhline(y=y_calc, c='r', ls=':', label=label)
            ax.grid()
            ax.set_title('Average FP-Probability of different code word lengths k', fontdict=fontdict)
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('Code word length k', fontsize=8)
            ax.legend()

            # plt.savefig(save_k)
            # np.save(save_k, prob_data_2D)
            plt.show()

        elif plot_what == 'z':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            bar_width = 0.8 / x_r
            ax.bar(x_c, mean_c, width=0.9*bar_width)
            ax.axhline(y=y_calc, c='r', ls=':', label=label)
            ax.set_title('Average FP-Probability of different symbol sizes z', fontdict=fontdict)
            ax.grid()
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('Symbol Size z', fontsize=8)
            ax.legend()

            # plt.savefig(save_z)
            # np.save(save_z, prob_data_2D)

            plt.show()

        elif plot_what == 'q':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            bar_width = 0.8 / x_r
            ax.bar(x_c, mean_c, width=0.9*bar_width)
            ax.axhline(y=y_calc, c='r', ls=':', label=label)
            ax.set_title('Average FP-Probability of different q-ary bases q', fontdict=fontdict)
            ax.grid()
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('q-ary bases q', fontsize=8)
            ax.legend()

            # plt.savefig(save_q)
            # np.save(save_q, prob_data_2D)

            plt.show()

        # mean_prob = np.array([mean_r, mean_c]) # if needed for further actions

        # return mean_prob # if needed for further actions

    def plot_prob_multiple(self, data_rs, data_sha1, data_sha2, data_nc, plot_what):#
        alg = self.alg
        n = self.n
        z = self.z
        labels_k = self.convert_n_to_k(n)
        z_r = self.convert_z_correct(z)
        labels_q = self.convert_z_to_q(z_r)
        title = ''
        save1 = ''  # file name
        save2 = ''
        fontdict = {'fontsize': 9}  # font size of titles
        y_label = ''

        # if-clause for checking for equal dimensions
        num_rows = data_rs.shape[0]
        num_columns = data_rs.shape[1]

        #x_r = np.arange(num_rows) + 1
        x_r = np.array(self.convert_z_correct(z))
        x_c = np.arange(num_columns) + 1

        if plot_what == 'z':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            bar_width = 0.2 / num_rows
            for i in range(num_rows):
                bar_position = x_c + i * bar_width
                ax.bar(bar_position - bar_width, data_rs[i, :], width=0.9 * bar_width, label='RS')
                ax.bar(bar_position, data_sha1[i, :], width=0.9 * bar_width, label='SHA1')
                ax.bar(bar_position + bar_width, data_sha2[i, :], width=0.9 * bar_width, label='SHA2')
                ax.bar(bar_position + 2 * bar_width, data_nc[i, :], width=0.9 * bar_width, label='NC')

            ax.set_title('FP-Probability-Graphs for different symbol sizes z', fontdict=fontdict)
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('Code length n', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticks(x_c + (num_rows - 1) * bar_width / 2)
            ax.set_xticklabels(x_c)

            # plt.savefig(save1)
            # np.save(save1, prob_data_2D)
            plt.show()

        elif plot_what == 'q':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            bar_width = 0.2 / num_rows
            for i in range(num_rows):
                bar_position = x_c + i * bar_width
                ax.bar(bar_position - bar_width, data_rs[i, :], width=0.9 * bar_width, label='RS')
                ax.bar(bar_position, data_sha1[i, :], width=0.9 * bar_width, label='SHA1')
                ax.bar(bar_position + bar_width, data_sha2[i, :], width=0.9 * bar_width, label='SHA2')
                ax.bar(bar_position + 2 * bar_width, data_nc[i, :], width=0.9 * bar_width, label='NC')

            ax.set_title('FP-Probability-Graphs for different symbol sizes z', fontdict=fontdict)
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('Code word length k', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticks(x_c + (num_rows - 1) * bar_width / 2)
            ax.set_xticklabels(labels_k)

            # plt.savefig(save1)
            # np.save(save1, prob_data_2D)
            plt.show()

        # elif plot_what == 'n':
        #     fig, ax = plt.subplots()
        #     fig.suptitle(title, fontsize=14)
        #
        #     bar_width = 0.2 / num_columns
        #     print(num_columns)
        #     for j in range(num_columns):
        #         bar_position = x_r + j * bar_width
        #         ax.bar(bar_position - bar_width, data_rs[:, j], width=0.9 * bar_width, c='black', label='RS')
        #         ax.bar(bar_position, data_sha1[:, j], width=0.9 * bar_width,  c='yellow', label='SHA1')
        #         # ax.bar(bar_position + bar_width, data_sha2[:, j], width=0.9 * bar_width,  c='green', label='SHA2')
        #         # ax.bar(bar_position + 2 * bar_width, data_nc[:, j], width=0.9 * bar_width, c='magenta', label='NC')
        #
        #     ax.set_title('FP-Probability-Graphs for different code lengths n', fontdict=fontdict)
        #     ax.set_ylabel(y_label, fontsize=8)
        #     ax.set_xlabel('Symbol size z', fontsize=8)
        #     ax.grid()
        #     ax.legend()
        #     ax.set_xticks(x_r + (num_columns - 1) * bar_width / 2)
        #     ax.set_xticklabels(x_r)
        #
        #     # plt.savefig(save2)
        #     # np.save(save2, prob_data_2D)
        #
        #     plt.show()

    def boxplot_prob(self, prob_data_2D, plot_what):
        alg = self.alg
        n = self.n
        z = self.z
        labels_k = self.convert_n_to_k(n)
        z_r = self.convert_z_correct(z)
        labels_q = self.convert_z_to_q(z_r)
        title = ''
        save_z = ''  # file name
        save_n = ''
        save_q = ''
        save_k = ''
        label = ''
        fontdict = {'fontsize': 9}  # font size of titles
        y_label = 'FP-Probability'

        if alg == 'rs':
            title = 'FP-Probability of Reed Solomon'
            save_z = 'RS_Prob_SameZ_DiffN.png'
            save_n = 'RS_Prob_SameN_DiffZ.png'
            save_q = 'RS_Prob_SameQ_DiffK.png'
            save_k = 'RS_Prob_SameK_DiffQ.png'
            label = 'Calculated FP-Probability RS'
            y_calc = 0.004 #@Ed richtigen wert bitte einfügen
        elif alg == 'sha1':
            title = 'FP-Probability of Sha1'
            save_z = 'SHA1_Prob_SameZ_DiffN.png'
            save_n = 'SHA1_Prob_SameN_DiffZ.png'
            save_q = 'SHA1_Prob_SameQ_DiffK.png'
            save_k = 'SHA1_Prob_SameK_DiffQ.png'
            label = 'Calculated FP-Probability SHA1'
            y_calc = 0.004 #@Ed richtigen wert bitte einfügen
        elif alg == 'sha2':
            title = 'FP-Probability of Sha2'
            save_z = 'SHA2_Prob_SameZ_DiffN.png'
            save_n = 'SHA2_Prob_SameN_DiffZ.png'
            save_q = 'SHA2_Prob_SameQ_DiffK.png'
            save_k = 'SHA2_Prob_SameK_DiffQ.png'
            label = 'Calculated FP-Probability SHA2'
            y_calc = 0.004 #@Ed richtigen wert bitte einfügen
        elif alg == 'nc':
            title = 'FP-Probability of No Code'
            save_z = 'NC_Prob_SameZ_DiffN.png'
            save_n = 'NC_Prob_SameN_DiffZ.png'
            save_q = 'NC_Prob_SameQ_DiffK.png'
            save_k = 'NC_Prob_SameK_DiffQ.png'
            label = 'Calculated FP-Probability NC'
            y_calc = 0.004 #@Ed richtigen wert bitte einfügen

        num_columns = prob_data_2D.shape[1]

        x_r = np.array(self.convert_z_correct(z))
        x_c = np.arange(num_columns) + 1

        if plot_what == 'z':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            ax.boxplot(prob_data_2D, patch_artist=True,
                showmeans=False, showfliers=False,
                medianprops={"color": "red", "linewidth": 0.5},
                boxprops={"facecolor": "C0", "edgecolor": "white",
                          "linewidth": 0.5},
                whiskerprops={"color": "C0", "linewidth": 1.5},
                capprops={"color": "C0", "linewidth": 1.5})
            
            ax.axhline(y=y_calc, c='r', ls=':', label=label)
            subtitle = 'FP-Probability for different symbol sizes z = ' + ', '.join(['{}'.format(b) for b in x_r])
            ax.set_title(subtitle, fontdict=fontdict)
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('n with k = 2^n', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticklabels(x_c)

            # plt.savefig(save_z)
            # np.save(save_z, prob_data_2D)
            plt.show()

        elif plot_what == 'q':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            ax.boxplot(prob_data_2D, patch_artist=True,
                       showmeans=False, showfliers=False,
                       medianprops={"color": "red", "linewidth": 0.5},
                       boxprops={"facecolor": "C0", "edgecolor": "white",
                                 "linewidth": 0.5},
                       whiskerprops={"color": "C0", "linewidth": 1.5},
                       capprops={"color": "C0", "linewidth": 1.5})

            ax.axhline(y=y_calc, c='r', ls=':', label=label)
            suptitel = 'FP-Probability for different q-ary bases q = ' + ', '.join(['2^{}'.format(b) for b in x_r])
            ax.set_title(suptitel, fontdict=fontdict)
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('Code word length k', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticklabels(['$2^{{{}}}$'.format(b) for b in x_c])  # = labels_k

            # plt.savefig(save_q)
            # np.save(save_q, prob_data_2D)
            plt.show()

        elif plot_what == 'n':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            ax.boxplot(prob_data_2D.T, patch_artist=True,
                       showmeans=False, showfliers=False,
                       medianprops={"color": "red", "linewidth": 0.5},
                       boxprops={"facecolor": "C0", "edgecolor": "white",
                                 "linewidth": 0.5},
                       whiskerprops={"color": "C0", "linewidth": 1.5},
                       capprops={"color": "C0", "linewidth": 1.5})

            ax.axhline(y=y_calc, c='r', ls=':', label=label)
            subtitle = 'FP-Probability for different n = ' + ', '.join(['{}'.format(b) for b in x_c])
            ax.set_title(subtitle, fontdict=fontdict)
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('Symbol size z', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticklabels(x_r)

            # plt.savefig(save_n)
            # np.save(save_n, prob_data_2D)

            plt.show()

        elif plot_what == 'k':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            ax.boxplot(prob_data_2D.T, patch_artist=True,
                       showmeans=False, showfliers=False,
                       medianprops={"color": "red", "linewidth": 0.5},
                       boxprops={"facecolor": "C0", "edgecolor": "white",
                                 "linewidth": 0.5},
                       whiskerprops={"color": "C0", "linewidth": 1.5},
                       capprops={"color": "C0", "linewidth": 1.5})

            ax.axhline(y=y_calc, c='r', ls=':', label=label)
            subtitle = "FP-Probability for different code word lengths k = " + ', '.join(['2^{}'.format(b) for b in x_c])
            ax.set_title(subtitle, fontdict=fontdict)
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('q-ary basis q', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticklabels(['$2^{{{}}}$'.format(b) for b in x_r])

            # plt.savefig(save_k)
            # np.save(save_k, prob_data_2D)

            plt.show()

    def violinplot_prob(self, prob_data_2D, plot_what):
        alg = self.alg
        n = self.n
        z = self.z
        labels_k = self.convert_n_to_k(n)
        z_r = self.convert_z_correct(z)
        labels_q = self.convert_z_to_q(z_r)
        title = ''
        save_z = ''  # file name
        save_n = ''
        save_q = ''
        save_k = ''
        label = ''
        fontdict = {'fontsize': 9}  # font size of titles
        y_label = 'FP-Probability'

        if alg == 'rs':
            title = 'FP-Probability of Reed Solomon'
            save_z = 'RS_Prob_SameZ_DiffN.png'
            save_n = 'RS_Prob_SameN_DiffZ.png'
            save_q = 'RS_Prob_SameQ_DiffK.png'
            save_k = 'RS_Prob_SameK_DiffQ.png'
            label = 'Calculated FP-Probability RS'
            y_calc = 0.004  # @Ed richtigen wert bitte einfügen
        elif alg == 'sha1':
            title = 'FP-Probability of Sha1'
            save_z = 'SHA1_Prob_SameZ_DiffN.png'
            save_n = 'SHA1_Prob_SameN_DiffZ.png'
            save_q = 'SHA1_Prob_SameQ_DiffK.png'
            save_k = 'SHA1_Prob_SameK_DiffQ.png'
            label = 'Calculated FP-Probability SHA1'
            y_calc = 0.004  # @Ed richtigen wert bitte einfügen
        elif alg == 'sha2':
            title = 'FP-Probability of Sha2'
            save_z = 'SHA2_Prob_SameZ_DiffN.png'
            save_n = 'SHA2_Prob_SameN_DiffZ.png'
            save_q = 'SHA2_Prob_SameQ_DiffK.png'
            save_k = 'SHA2_Prob_SameK_DiffQ.png'
            label = 'Calculated FP-Probability SHA2'
            y_calc = 0.004  # @Ed richtigen wert bitte einfügen
        elif alg == 'nc':
            title = 'FP-Probability of No Code'
            save_z = 'NC_Prob_SameZ_DiffN.png'
            save_n = 'NC_Prob_SameN_DiffZ.png'
            save_q = 'NC_Prob_SameQ_DiffK.png'
            save_k = 'NC_Prob_SameK_DiffQ.png'
            label = 'Calculated FP-Probability NC'
            y_calc = 0.004  # @Ed richtigen wert bitte einfügen

        num_columns = prob_data_2D.shape[1]

        x_r = np.array(self.convert_z_correct(z))
        x_c = np.arange(num_columns) + 1

        if plot_what == 'z':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            ax.violinplot(prob_data_2D, showmeans=True, showmedians=False, showextrema=True)

            ax.axhline(y=y_calc, c='r', ls=':', label=label)
            subtitle = 'FP-Probability for different symbol sizes z = ' + ', '.join(['{}'.format(b) for b in x_r])
            ax.set_title(subtitle, fontdict=fontdict)
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('n with k = 2^n', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticklabels(x_c)

            # plt.savefig(save_z)
            # np.save(save_z, prob_data_2D)
            plt.show()

        elif plot_what == 'q':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            ax.violinplot(prob_data_2D, showmeans=True, showmedians=False, showextrema=True)

            ax.axhline(y=y_calc, c='r', ls=':', label=label)
            suptitel = 'FP-Probability for different q-ary bases q = ' + ', '.join(['2^{}'.format(b) for b in x_r])
            ax.set_title(suptitel, fontdict=fontdict)
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('Code word length k', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticklabels(['$2^{{{}}}$'.format(b) for b in x_c])  # = labels_k

            # plt.savefig(save_q)
            # np.save(save_q, prob_data_2D)
            plt.show()

        elif plot_what == 'n':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            ax.violinplot(prob_data_2D.T, showmeans=True, showmedians=False, showextrema=True)

            ax.axhline(y=y_calc, c='r', ls=':', label=label)
            subtitle = 'FP-Probability for different n = ' + ', '.join(['{}'.format(b) for b in x_c])
            ax.set_title(subtitle, fontdict=fontdict)
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('Symbol size z', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticklabels(x_r)

            # plt.savefig(save_n)
            # np.save(save_n, prob_data_2D)

            plt.show()

        elif plot_what == 'k':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            ax.violinplot(prob_data_2D.T, showmeans=True, showmedians=False, showextrema=True)

            ax.axhline(y=y_calc, c='r', ls=':', label=label)
            subtitle = "FP-Probability for different code word lengths k = " + ', '.join(
                ['2^{}'.format(b) for b in x_c])
            ax.set_title(subtitle, fontdict=fontdict)
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('q-ary basis q', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticklabels(['$2^{{{}}}$'.format(b) for b in x_r])

            # plt.savefig(save_k)
            # np.save(save_k, prob_data_2D)

            plt.show()