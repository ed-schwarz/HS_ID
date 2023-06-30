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

    def plot_prob_multiple(self, data_rs, data_sha1, data_sha2, data_nc, plot_what):
        n = self.n
        z = self.z
        labels_k = self.convert_n_to_k(n)
        z_r = self.convert_z_correct(z)
        labels_q = self.convert_z_to_q(z_r)
        title = 'FP-Probability for RS, SHA1, SHA2 and NC'
        save = ''  # file name
        fontdict = {'fontsize': 9}  # font size of titles
        y_label = 'FP-Probability'
        y_calc = 0.04
        label = 'Calculated FP-Probability'

        # if-clause for checking for equal dimensions
        num_rows = data_rs.shape[0]
        num_columns = data_rs.shape[1]

        x_r = np.array(self.convert_z_correct(z))
        x_c = np.arange(num_columns) + 1

        if plot_what == 'z':
            for i in range(num_rows):
                data_rs = np.multiply(data_rs, 10)
                data_sha1 = np.multiply(data_sha1, 10)
                data_sha2 = np.multiply(data_sha2, 10)
                data_nc = np.multiply(data_nc, 10)
                fig, ax = plt.subplots()
                fig.suptitle(title, fontsize=14)
                bar_width = 0.2 / num_rows
                ax.bar(x_c - bar_width, data_rs[i, :], width=0.9 * bar_width, label='RS')
                ax.bar(x_c, data_sha1[i, :], width=0.9 * bar_width, label='SHA1')
                ax.bar(x_c + bar_width, data_sha2[i, :], width=0.9 * bar_width, label='SHA2')
                ax.bar(x_c + 2 * bar_width, data_nc[i, :], width=0.9 * bar_width, label='NC')
                ax.axhline(y=y_calc, c='r', ls=':', label=label)

                ax.set_title('FP-Probability for the symbol sizes z = {}'.format(x_r[i]), fontdict=fontdict)
                ax.set_ylabel(y_label, fontsize=8)
                ax.set_xlabel('n with k = 2^n', fontsize=8)
                ax.grid()
                ax.legend()
                ax.set_xticks(x_c + (num_rows - 1) * bar_width / 2)
                ax.set_xticklabels(x_c)

                save = 'plot_prob_multiple_all_z={}.png'.format(x_r[i])
                plt.savefig(save)
                plt.show()

        # np.save(save, prob_data_2D)

        elif plot_what == 'q':
            for i in range(num_rows):
                fig, ax = plt.subplots()
                fig.suptitle(title, fontsize=14)
                bar_width = 0.2 / num_rows
                ax.bar(x_c - bar_width, data_rs[i, :], width=0.9 * bar_width, label='RS')
                ax.bar(x_c, data_sha1[i, :], width=0.9 * bar_width, label='SHA1')
                ax.bar(x_c + bar_width, data_sha2[i, :], width=0.9 * bar_width, label='SHA2')
                ax.bar(x_c + 2 * bar_width, data_nc[i, :], width=0.9 * bar_width, label='NC')
                ax.axhline(y=y_calc, c='r', ls=':', label=label)


                ax.set_title('FP-Probability for q-ary basis q = 2^{}'.format(x_r[i]), fontdict=fontdict)
                ax.set_ylabel(y_label, fontsize=8)
                ax.set_xlabel('Code word length k', fontsize=8)
                ax.grid()
                ax.legend()
                ax.set_xticks(x_c + (num_rows - 1) * bar_width / 2)
                ax.set_xticklabels(['$2^{{{}}}$'.format(b) for b in x_c])  # = labels_k

                save = 'plot_prob_multiple_all_q=2^{}.png'.format(x_r[i])
                plt.savefig(save)
                plt.show()

        # np.save(save, prob_data_2D)

        elif plot_what == 'n':
            for j in range(num_columns):
                fig, ax = plt.subplots()
                fig.suptitle(title, fontsize=14)
                bar_width = 0.8 / num_rows
                ax.bar(x_r - bar_width, data_rs[:, j], width=0.9 * bar_width, label='RS')
                ax.bar(x_r, data_sha1[:, j], width=0.9 * bar_width, label='SHA1')
                ax.bar(x_r + bar_width, data_sha2[:, j], width=0.9 * bar_width, label='SHA2')
                ax.bar(x_r + 2 * bar_width, data_nc[:, j], width=0.9 * bar_width, label='NC')
                ax.axhline(y=y_calc, c='r', ls=':', label=label)

                ax.set_title('FP-Probability for different code lengths n = {}'.format(x_c[j]), fontdict=fontdict)
                ax.set_ylabel(y_label, fontsize=8)
                ax.set_xlabel('Symbol size z', fontsize=8)
                ax.grid()
                ax.legend()
                ax.set_xticks(x_r + (num_columns - 1) * bar_width / 2)
                ax.set_xticklabels(x_r)

                save = 'plot_prob_multiple_all_n={}.png'.format(x_c[j])
                plt.savefig(save)
                plt.show()

            # np.save(save, prob_data_2D)

        elif plot_what == 'k':
            for j in range(num_columns):
                fig, ax = plt.subplots()
                fig.suptitle(title, fontsize=14)
                bar_width = 0.8 / num_rows
                ax.bar(x_r - bar_width, data_rs[:, j], width=0.9 * bar_width, label='RS')
                ax.bar(x_r, data_sha1[:, j], width=0.9 * bar_width, label='SHA1')
                ax.bar(x_r + bar_width, data_sha2[:, j], width=0.9 * bar_width, label='SHA2')
                ax.bar(x_r + 2 * bar_width, data_nc[:, j], width=0.9 * bar_width, label='NC')
                ax.axhline(y=y_calc, c='r', ls=':', label=label)

                ax.set_title('FP-Probability for code word lengths k = 2^{}'.format(x_c[j]), fontdict=fontdict)
                ax.set_ylabel(y_label, fontsize=8)
                ax.set_xlabel('q-ary basis q', fontsize=8)
                ax.grid()
                ax.legend()
                ax.set_xticks(x_r + (num_columns - 1) * bar_width / 2)
                ax.set_xticklabels(['$2^{{{}}}$'.format(b) for b in x_r])

                save = 'plot_prob_multiple_all_k=2^{}.png'.format(x_c[j])
                plt.savefig(save)
                plt.show()

                # np.save(save, prob_data_2D)

    def bp_vp_prob_3D(self, prob_data_3D, plot_what):
        alg = self.alg
        n = self.n
        z = self.z
        labels_k = self.convert_n_to_k(n)
        z_r = self.convert_z_correct(z)
        labels_q = self.convert_z_to_q(z_r)
        title_bp = ''
        title_vp = ''
        save_z = ''  # file name
        save_n = ''
        save_q = ''
        save_k = ''
        label = ''
        fontdict = {'fontsize': 9}  # font size of titles
        y_label = 'FP-Probability'

        if alg == 'rs':
            title_bp = 'Boxplot FP-Probability of Reed Solomon'
            title_vp = 'Violinplot FP-Probability of Reed Solomon'
            save_z = 'RS_Prob_SameZ_DiffN.png'
            save_n = 'RS_Prob_SameN_DiffZ.png'
            save_q = 'RS_Prob_SameQ_DiffK.png'
            save_k = 'RS_Prob_SameK_DiffQ.png'
            label = 'Calculated FP-Probability RS'
            y_calc = 0.004 #@Ed richtigen wert bitte einfügen
        elif alg == 'sha1':
            title_bp = 'Boxplot FP-Probability of Sha1'
            title_vp = 'Violinplot FP-Probability of Sha1'
            save_z = 'SHA1_Prob_SameZ_DiffN.png'
            save_n = 'SHA1_Prob_SameN_DiffZ.png'
            save_q = 'SHA1_Prob_SameQ_DiffK.png'
            save_k = 'SHA1_Prob_SameK_DiffQ.png'
            label = 'Calculated FP-Probability SHA1'
            y_calc = 0.004 #@Ed richtigen wert bitte einfügen
        elif alg == 'sha2':
            title_bp = 'Boxplot FP-Probability of Sha2'
            title_vp = 'Violinplot FP-Probability of Sha2'
            save_z = 'SHA2_Prob_SameZ_DiffN.png'
            save_n = 'SHA2_Prob_SameN_DiffZ.png'
            save_q = 'SHA2_Prob_SameQ_DiffK.png'
            save_k = 'SHA2_Prob_SameK_DiffQ.png'
            label = 'Calculated FP-Probability SHA2'
            y_calc = 0.004 #@Ed richtigen wert bitte einfügen
        elif alg == 'nc':
            title_bp = 'Boxplot FP-Probability of No Code'
            title_vp = 'Violinplot FP-Probability of No Code'
            save_z = 'NC_Prob_SameZ_DiffN.png'
            save_n = 'NC_Prob_SameN_DiffZ.png'
            save_q = 'NC_Prob_SameQ_DiffK.png'
            save_k = 'NC_Prob_SameK_DiffQ.png'
            label = 'Calculated FP-Probability NC'
            y_calc = 0.004 #@Ed richtigen wert bitte einfügen

        num_rows = prob_data_3D.shape[0]
        num_columns = prob_data_3D.shape[1]
        num_elements = prob_data_3D.shape[2]

        x_r = np.array(self.convert_z_correct(z))
        x_c = np.arange(num_columns) + 1

        if plot_what == 'z':
            arr2D = np.zeros([num_columns, num_elements])
            for i in range(num_rows):
                for j in range(num_columns):
                    for k in range(num_elements):
                        arr2D[j, k] = prob_data_3D[i, j, k]
                        print("i = {ay}, j = {jay}, k = {kay}".format(ay=i, jay=j, kay=k))

                fig, ax = plt.subplots()
                fig.suptitle(title_bp, fontsize=14)

                ax.boxplot(arr2D.T, patch_artist=True,
                    showmeans=False, showfliers=False,
                    medianprops={"color": "red", "linewidth": 0.5},
                    boxprops={"facecolor": "C0", "edgecolor": "white",
                              "linewidth": 0.5},
                    whiskerprops={"color": "C0", "linewidth": 1.5},
                    capprops={"color": "C0", "linewidth": 1.5})

                #ax.axhline(y=y_calc, c='r', ls=':', label=label)
                subtitle = 'FP-Probability for different symbol sizes z = ' + ', '.join(['{}'.format(x_r[i])])
                ax.set_title(subtitle, fontdict=fontdict)
                ax.set_ylabel(y_label, fontsize=8)
                ax.set_xlabel('n with k = 2^n', fontsize=8)
                ax.set_ylim(0, 0.0125)
                ax.grid()
                #ax.legend()
                ax.set_xticklabels(x_c)

                # plt.savefig(save_z)
                # np.save(save_z, arr2D)
                plt.show()
                print('Boxplot plotted for i = {}'.format(i))

                fig, ax = plt.subplots()
                fig.suptitle(title_vp, fontsize=14)

                ax.violinplot(arr2D.T, showmeans=True, showmedians=False, showextrema=False)
                #ax.axhline(y=y_calc, c='r', ls=':', label=label)
                subtitle = 'FP-Probability for different symbol sizes z = ' + ', '.join(['{}'.format(x_r[i])])
                ax.set_title(subtitle, fontdict=fontdict)
                ax.set_ylabel(y_label, fontsize=8)
                ax.set_xlabel('n with k = 2^n', fontsize=8)
                ax.set_ylim(0, 0.0125)
                ax.grid()
                #ax.legend()
                ax.set_xticklabels(x_c)

                # plt.savefig(save_z)
                # np.save(save_z, arr2D)
                plt.show()
                print('ViolinPlot plotted for i = {}'.format(i))

        elif plot_what == 'q':
            arr2D = np.zeros([num_columns, num_elements])
            for i in range(num_rows):
                for j in range(num_columns):
                    for k in range(num_elements):
                        arr2D[j, k] = prob_data_3D[i, j, k]
                        print("i = {ay}, j = {jay}, k = {kay}".format(ay=i, jay=j, kay=k))

                fig, ax = plt.subplots()
                fig.suptitle(title_bp, fontsize=14)

                ax.boxplot(arr2D.T, patch_artist=True,
                           showmeans=False, showfliers=False,
                           medianprops={"color": "red", "linewidth": 0.5},
                           boxprops={"facecolor": "C0", "edgecolor": "white",
                                     "linewidth": 0.5},
                           whiskerprops={"color": "C0", "linewidth": 1.5},
                           capprops={"color": "C0", "linewidth": 1.5})

                #ax.axhline(y=y_calc, c='r', ls=':', label=label)
                suptitel = 'FP-Probability for different q-ary bases q = ' + ', '.join(['2^{}'.format(b) for b in x_r])
                ax.set_title(suptitel, fontdict=fontdict)
                ax.set_ylabel(y_label, fontsize=8)
                ax.set_xlabel('Code word length k', fontsize=8)
                ax.set_ylim(0, 0.0125)
                ax.grid()
                #ax.legend()
                ax.set_xticklabels(['$2^{{{}}}$'.format(b) for b in x_c])  # = labels_k

                # plt.savefig(save_q)
                # np.save(save_q, prob_data_2D)
                plt.show()
                print('Boxplot plotted for i = {}'.format(i))

                fig, ax = plt.subplots()
                fig.suptitle(title_vp, fontsize=14)

                ax.violinplot(arr2D.T, showmeans=True, showmedians=False, showextrema=False)
                # ax.axhline(y=y_calc, c='r', ls=':', label=label)
                suptitel = 'FP-Probability for different q-ary bases q = ' + ', '.join(['2^{}'.format(b) for b in x_r])
                ax.set_title(suptitel, fontdict=fontdict)
                ax.set_ylabel(y_label, fontsize=8)
                ax.set_xlabel('Code word length k', fontsize=8)
                ax.set_ylim(0, 0.0125)
                ax.grid()
                # ax.legend()
                ax.set_xticklabels(['$2^{{{}}}$'.format(b) for b in x_c])  # = labels_k

                # plt.savefig(save_q)
                # np.save(save_q, prob_data_2D)
                plt.show()
                print('ViolinPlot plotted for i = {}'.format(i))

        elif plot_what == 'n':
            arr2D = np.zeros([num_rows, num_elements])
            for i in range(num_columns):
                for j in range(num_rows):
                    for k in range(num_elements):
                        arr2D[j, k] = prob_data_3D[j, i, k]
                        print("i = {ay}, j = {jay}, k = {kay}".format(ay=i, jay=j, kay=k))

                fig, ax = plt.subplots()
                fig.suptitle(title_bp, fontsize=14)

                ax.boxplot(arr2D.T, patch_artist=True,
                           showmeans=False, showfliers=False,
                           medianprops={"color": "red", "linewidth": 0.5},
                           boxprops={"facecolor": "C0", "edgecolor": "white",
                                     "linewidth": 0.5},
                           whiskerprops={"color": "C0", "linewidth": 1.5},
                           capprops={"color": "C0", "linewidth": 1.5})

                #ax.axhline(y=y_calc, c='r', ls=':', label=label)
                subtitle = 'FP-Probability for different n = ' + ', '.join(['{}'.format(b) for b in x_c])
                ax.set_title(subtitle, fontdict=fontdict)
                ax.set_ylabel(y_label, fontsize=8)
                ax.set_xlabel('Symbol size z', fontsize=8)
                ax.set_ylim(0, 0.0125)
                ax.grid()
                #ax.legend()
                ax.set_xticklabels(x_r)

                # plt.savefig(save_n)
                # np.save(save_n, prob_data_2D)
                plt.show()
                print('Boxplot plotted for i = {}'.format(i))

                fig, ax = plt.subplots()
                fig.suptitle(title_vp, fontsize=14)

                ax.violinplot(arr2D.T, showmeans=True, showmedians=False, showextrema=False)
                # ax.axhline(y=y_calc, c='r', ls=':', label=label)
                subtitle = 'FP-Probability for different n = ' + ', '.join(['{}'.format(b) for b in x_c])
                ax.set_title(subtitle, fontdict=fontdict)
                ax.set_ylabel(y_label, fontsize=8)
                ax.set_xlabel('Symbol size z', fontsize=8)
                ax.set_ylim(0, 0.0125)
                ax.grid()
                # ax.legend()
                ax.set_xticklabels(x_r)

                # plt.savefig(save_n)
                # np.save(save_n, prob_data_2D)
                plt.show()
                print('ViolinPlot plotted for i = {}'.format(i))


        elif plot_what == 'k':
            arr2D = np.zeros([num_rows, num_elements])
            for i in range(num_columns):
                for j in range(num_rows):
                    for k in range(num_elements):
                        arr2D[j, k] = prob_data_3D[j, i, k]
                        print("i = {ay}, j = {jay}, k = {kay}".format(ay=i, jay=j, kay=k))

            fig, ax = plt.subplots()
            fig.suptitle(title_bp, fontsize=14)

            ax.boxplot(arr2D.T, patch_artist=True,
                       showmeans=False, showfliers=False,
                       medianprops={"color": "red", "linewidth": 0.5},
                       boxprops={"facecolor": "C0", "edgecolor": "white",
                                 "linewidth": 0.5},
                       whiskerprops={"color": "C0", "linewidth": 1.5},
                       capprops={"color": "C0", "linewidth": 1.5})

            #ax.axhline(y=y_calc, c='r', ls=':', label=label)
            subtitle = "FP-Probability for different code word lengths k = " + ', '.join(['2^{}'.format(b) for b in x_c])
            ax.set_title(subtitle, fontdict=fontdict)
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('q-ary basis q', fontsize=8)
            ax.set_ylim(0, 0.0125)
            ax.grid()
            #ax.legend()
            ax.set_xticklabels(['$2^{{{}}}$'.format(b) for b in x_r])

            # plt.savefig(save_k)
            # np.save(save_k, prob_data_2D)
            plt.show()
            print('Boxplot plotted for i = {}'.format(i))

            fig, ax = plt.subplots()
            fig.suptitle(title_vp, fontsize=14)

            ax.violinplot(arr2D.T, showmeans=True, showmedians=False, showextrema=False)
            # ax.axhline(y=y_calc, c='r', ls=':', label=label)
            subtitle = "FP-Probability for different code word lengths k = " + ', '.join(
                ['2^{}'.format(b) for b in x_c])
            ax.set_title(subtitle, fontdict=fontdict)
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('q-ary basis q', fontsize=8)
            ax.set_ylim(0, 0.0125)
            ax.grid()
            # ax.legend()
            ax.set_xticklabels(['$2^{{{}}}$'.format(b) for b in x_r])

            # plt.savefig(save_k)
            # np.save(save_k, prob_data_2D)
            plt.show()
            print('Boxplot plotted for i = {}'.format(i))

    def bp_vp_prob_1D(self, prob_data_1D, z_i, n_i): #z_i, n_i used for generating 1D-Array
        #z_i equals 8, 16, 32. NOT Index in z*n-Array

        alg = self.alg
        #n = self.n
        #z = self.z
        #labels_k = self.convert_n_to_k(n_i)
        z_r = self.convert_z_correct(z_i)  # convert index into actual z value
        z = z_r[0]
        #labels_q = self.convert_z_to_q(z_r)
        title = 'z = {z}, n = {n}'.format(z=z, n=n_i)
        label_calc = 'FP_avg_calc =  1/q = 1/(2^z)'
        y_label = 'FP-Probability'
        y_calc = 1 / np.power(2, z)

        if alg == 'rs':
            print('RS Average: {}'.format(np.average(prob_data_1D[0, :])))
            fig, ax = plt.subplots()
            fig.suptitle('Boxplot FP-Probability of RS')
            ax.boxplot(prob_data_1D[0, :], patch_artist=True,
                       showmeans=False, showfliers=False,
                       medianprops={"color": "red", "linewidth": 0.5},
                       boxprops={"facecolor": "C0", "edgecolor": "white",
                                 "linewidth": 0.5},
                       whiskerprops={"color": "C0", "linewidth": 1.5},
                       capprops={"color": "C0", "linewidth": 1.5})
            ax.set_title(title)
            ax.axhline(y=y_calc, c='r', ls=':', label=label_calc)
            ax.set_ylabel(y_label)
            ax.legend()
            ax.grid()
            plt.show()

            fig, ax = plt.subplots()
            fig.suptitle('Violinplot FP-Probability of RS')
            ax.violinplot(prob_data_1D[0, :], showmeans=True, showmedians=False, showextrema=False)
            ax.set_title(title)
            ax.axhline(y=y_calc, c='r', ls=':', label=label_calc)
            ax.set_ylabel(y_label)
            ax.legend()
            ax.grid()
            plt.show()

        elif alg == 'sha1':
            print('Sha1 Average: {}'.format(np.average(prob_data_1D[0, :])))
            fig, ax = plt.subplots()
            fig.suptitle('Boxplot FP-Probability of Sha1')
            ax.boxplot(prob_data_1D[0, :], patch_artist=True,
                       showmeans=False, showfliers=False,
                       medianprops={"color": "red", "linewidth": 0.5},
                       boxprops={"facecolor": "C0", "edgecolor": "white",
                                 "linewidth": 0.5},
                       whiskerprops={"color": "C0", "linewidth": 1.5},
                       capprops={"color": "C0", "linewidth": 1.5})
            ax.set_title(title)
            ax.axhline(y=y_calc, c='r', ls=':', label=label_calc)
            ax.set_ylabel(y_label)
            ax.legend()
            ax.grid()
            plt.show()

            fig, ax = plt.subplots()
            fig.suptitle('Violinplot FP-Probability of Sha1')
            ax.violinplot(prob_data_1D[0, :], showmeans=True, showmedians=False, showextrema=False)
            ax.set_title(title)
            ax.axhline(y=y_calc, c='r', ls=':', label=label_calc)
            ax.set_ylabel(y_label)
            ax.legend()
            ax.grid()
            plt.show()

        elif alg == 'sha2':
            print('Sha2 Average: {}'.format(np.average(prob_data_1D[0, :])))
            fig, ax = plt.subplots()
            fig.suptitle('Boxplot FP-Probability of Sha2')
            ax.set_title(title)
            ax.boxplot(prob_data_1D[0, :], patch_artist=True,
                       showmeans=False, showfliers=False,
                       medianprops={"color": "red", "linewidth": 0.5},
                       boxprops={"facecolor": "C0", "edgecolor": "white",
                                 "linewidth": 0.5},
                       whiskerprops={"color": "C0", "linewidth": 1.5},
                       capprops={"color": "C0", "linewidth": 1.5})
            ax.axhline(y=y_calc, c='r', ls=':', label=label_calc)
            ax.set_ylabel(y_label)
            ax.legend()
            ax.grid()
            plt.show()

            fig, ax = plt.subplots()
            fig.suptitle('Violinplot FP-Probability of Sha2')
            ax.violinplot(prob_data_1D[0, :], showmeans=True, showmedians=False, showextrema=False)
            ax.set_title(title)
            ax.axhline(y=y_calc, c='r', ls=':', label=label_calc)
            ax.set_ylabel(y_label)
            ax.legend()
            ax.grid()
            plt.show()

        elif alg == 'nc':
            print('NC Average: {}'.format(np.average(prob_data_1D[0, :])))
            fig, ax = plt.subplots()
            fig.suptitle('Boxplot FP-Probability of NC')
            ax.set_title(title)
            ax.boxplot(prob_data_1D[0, :], patch_artist=True,
                       showmeans=False, showfliers=False,
                       medianprops={"color": "red", "linewidth": 0.5},
                       boxprops={"facecolor": "C0", "edgecolor": "white",
                                 "linewidth": 0.5},
                       whiskerprops={"color": "C0", "linewidth": 1.5},
                       capprops={"color": "C0", "linewidth": 1.5})
            ax.axhline(y=y_calc, c='r', ls=':', label=label_calc)
            ax.set_ylabel(y_label)
            ax.legend()
            ax.grid()
            plt.show()

            fig, ax = plt.subplots()
            fig.suptitle('Boxplot FP-Probability of NC')
            ax.set_title(title)
            ax.violinplot(prob_data_1D[0, :], showmeans=True, showmedians=False, showextrema=False)
            ax.axhline(y=y_calc, c='r', ls=':', label=label_calc)
            ax.set_ylabel(y_label)
            ax.legend()
            ax.grid()
            plt.show()

        print('Plots done')

    def plot_prob_var(self, prob_data_2D, plot_what):
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
        y_label = 'Variance'

        prob_var_rows = np.zeros([z, n])
        prob_var_cols = np.zeros([z, n])

        if alg == 'rs':
            prob_var_rows = np.var(prob_data_2D.T, axis=0)
            prob_var_cols = np.var(prob_data_2D, axis=0)
            title = 'Variance of FP-Probability of Reed Solomon'
            save_z = 'RS_Prob_Var_SameZ_DiffN.png'
            save_n = 'RS_Prob_Var_SameN_DiffZ.png'
            save_q = 'RS_Prob_Var_SameQ_DiffK.png'
            save_k = 'RS_Prob_Var_SameK_DiffQ.png'
        elif alg == 'sha1':
            prob_var_rows = np.var(prob_data_2D.T, axis=0)
            prob_var_cols = np.var(prob_data_2D, axis=0)
            title = 'Variance of FP-Probability of Sha1'
            save_z = 'SHA1_Prob_Var_SameZ_DiffN.png'
            save_n = 'SHA1_Prob_Var_SameN_DiffZ.png'
            save_q = 'SHA1_Prob_Var_SameQ_DiffK.png'
            save_k = 'SHA1_Prob_Var_SameK_DiffQ.png'
        elif alg == 'sha2':
            prob_var_rows = np.var(prob_data_2D.T, axis=0)
            prob_var_cols = np.var(prob_data_2D, axis=0)
            title = 'Variance of FP-Probability of Sha2'
            save_z = 'NC_Prob_Var_SameZ_DiffN.png'
            save_n = 'NC_Prob_Var_SameN_DiffZ.png'
            save_q = 'NC_Prob_Var_SameQ_DiffK.png'
            save_k = 'NC_Prob_Var_SameK_DiffQ.png'
            label = 'Calculated FP-Probability SHA2'
            y_calc = 0.004 #@Ed richtigen wert bitte einfügen
        elif alg == 'nc':
            prob_var_rows = np.var(prob_data_2D.T, axis=0)
            prob_var_cols = np.var(prob_data_2D, axis=0)
            title = 'Variance of FP-Probability of No Code'
            save_z = 'NC_Prob_Var_SameZ_DiffN.png'
            save_n = 'NC_Prob_Var_SameN_DiffZ.png'
            save_q = 'NC_Prob_Var_SameQ_DiffK.png'
            save_k = 'NC_Prob_Var_SameK_DiffQ.png'

        num_columns = prob_data_2D.shape[1]
        num_rows = prob_data_2D.shape[0]

        x_r = np.array(self.convert_z_correct(z))
        x_c = np.arange(num_columns) + 1

        if plot_what == 'z':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            bar_width = 0.8 / num_rows
            for i in range(num_rows):
                bar_position = x_c + i * bar_width
                ax.bar(bar_position, prob_var_rows[i], width=bar_width, label='Var: z = {}'.format(x_r[i]))
            ax.set_title('Variance for different symbol sizes z', fontdict=fontdict)
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('n with k = 2^n', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticks(x_c + (num_rows - 1) * bar_width / 2)
            ax.set_xticklabels(x_c)

            # plt.savefig(save_z)
            # np.save(save_z, prob_var_rows)
            plt.show()


        elif plot_what == 'q':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            bar_width = 0.8 / num_rows
            for i in range(num_rows):
                bar_position = x_c + i * bar_width
                ax.bar(bar_position, prob_var_rows[i], width=bar_width, label='q = 2^{}'.format(x_r[i]))
            ax.set_title('Variance for different q-ary bases', fontdict=fontdict)
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('Code word length k', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticks(x_c + (num_rows - 1) * bar_width / 2)
            ax.set_xticklabels(['$2^{{{}}}$'.format(b) for b in x_c])  # = labels_k

            # plt.savefig(save_q)
            # np.save(save_q, prob_var_rows)
            plt.show()

        elif plot_what == 'n':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            bar_width = 0.8 / num_columns
            for j in range(num_columns):
                bar_position = x_r + j * bar_width
                ax.bar(bar_position, prob_var_cols[j], width=bar_width, label='n = {}'.format(j+1))
            ax.set_title('Variance for different code lengths n', fontdict=fontdict)
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('Symbol size z', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticks(x_r + (num_columns - 1) * bar_width / 2)
            ax.set_xticklabels(x_r)

            # plt.savefig(save_n)
            # np.save(save_n, prob_var_cols)

            plt.show()

        elif plot_what == 'k':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            bar_width = 0.8 / num_columns
            for j in range(num_columns):
                bar_position = x_r + j * bar_width
                ax.bar(bar_position, prob_var_cols[j], width=bar_width, label='k = 2^{}'.format(j + 1))
            ax.set_title('Variance for different code word lengths k', fontdict=fontdict)
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('q-ary basis q', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticks(x_r + (num_columns - 1) * bar_width / 2)
            ax.set_xticklabels(['$2^{{{}}}$'.format(b) for b in x_r])

            # plt.savefig(save_k)
            # np.save(save_k, prob_var_cols)

            plt.show()

    def plot_prob_std(self, prob_data_2D, plot_what):
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
        y_label = 'Standard Deviation'
        prob_std_rows = np.zeros([z, n])
        prob_std_cols = np.zeros([z, n])

        if alg == 'rs':
            title = 'Standard Deviation of FP-Probability of Reed Solomon'
            prob_std_rows = np.std(prob_data_2D.T, axis=0)
            prob_std_cols = np.std(prob_data_2D, axis=0)
            save_z = 'RS_Prob_Std_SameZ_DiffN.png'
            save_n = 'RS_Prob_Std_SameN_DiffZ.png'
            save_q = 'RS_Prob_Std_SameQ_DiffK.png'
            save_k = 'RS_Prob_Std_SameK_DiffQ.png'
        elif alg == 'sha1':
            title = 'Standard Deviation of FP-Probability of Sha1'
            prob_std_rows = np.std(prob_data_2D.T, axis=0)
            prob_std_cols = np.std(prob_data_2D, axis=0)
            save_z = 'SHA1_Prob_Std_SameZ_DiffN.png'
            save_n = 'SHA1_Prob_Std_SameN_DiffZ.png'
            save_q = 'SHA1_Prob_Std_SameQ_DiffK.png'
            save_k = 'SHA1_Prob_Std_SameK_DiffQ.png'
        elif alg == 'sha2':
            title = 'Standard Deviation of FP-Probability of Sha2'
            prob_std_rows = np.std(prob_data_2D.T, axis=0)
            prob_std_cols = np.std(prob_data_2D, axis=0)
            save_z = 'SHA2_Prob_Std_SameZ_DiffN.png'
            save_n = 'SHA2_Prob_Std_SameN_DiffZ.png'
            save_q = 'SHA2_Prob_Std_SameQ_DiffK.png'
            save_k = 'SHA2_Prob_Std_SameK_DiffQ.png'
        elif alg == 'nc':
            title = 'Standard Deviation of FP-Probability of No Code'
            prob_std_rows = np.std(prob_data_2D.T, axis=0)
            prob_std_cols = np.std(prob_data_2D, axis=0)
            save_z = 'NC_Prob_Std_SameZ_DiffN.png'
            save_n = 'NC_Prob_Std_SameN_DiffZ.png'
            save_q = 'NC_Prob_Std_SameQ_DiffK.png'
            save_k = 'NC_Prob_Std_SameK_DiffQ.png'

        num_columns = prob_data_2D.shape[1]
        num_rows = prob_data_2D.shape[0]

        x_r = np.array(self.convert_z_correct(z))
        x_c = np.arange(num_columns) + 1

        if plot_what == 'z':
            # Standard Deviation
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            bar_width = 0.8 / num_rows
            print('rows_std: {}'.format(prob_std_rows))

            for i in range(num_rows):
                bar_position = x_c + i * bar_width
                ax.bar(bar_position, prob_std_rows[i], width=bar_width, label='Std: z = {}'.format(x_r[i]))
            ax.set_title('Standard Deviation for different symbol sizes z', fontdict=fontdict)
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('n with k = 2^n', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticks(x_c + (num_rows - 1) * bar_width / 2)
            ax.set_xticklabels(x_c)

            # plt.savefig(save_z)
            # np.save(save_z, prob_std_rows)
            plt.show()

        elif plot_what == 'q':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            bar_width = 0.8 / num_rows
            for i in range(num_rows):
                bar_position = x_c + i * bar_width
                ax.bar(bar_position, prob_std_rows[i], width=bar_width, label='q = 2^{}'.format(x_r[i]))
            ax.set_title('Standard Deviation for different q-ary bases', fontdict=fontdict)
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('Code word length k', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticks(x_c + (num_rows - 1) * bar_width / 2)
            ax.set_xticklabels(['$2^{{{}}}$'.format(b) for b in x_c])  # = labels_k

            # plt.savefig(save_q)
            # np.save(save_q, prob_std_rows)
            plt.show()

        elif plot_what == 'n':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            bar_width = 0.8 / num_columns
            for j in range(num_columns):
                bar_position = x_r + j * bar_width
                ax.bar(bar_position, prob_std_cols[j], width=bar_width, label='n = {}'.format(j+1))
            ax.set_title('Standard Deviation for different code lengths n', fontdict=fontdict)
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('Symbol size z', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticks(x_r + (num_columns - 1) * bar_width / 2)
            ax.set_xticklabels(x_r)

            # plt.savefig(save_n)
            # np.save(save_n, prob_std_cols)

            plt.show()

        elif plot_what == 'k':
            fig, ax = plt.subplots()
            fig.suptitle(title, fontsize=14)

            bar_width = 0.8 / num_columns
            for j in range(num_columns):
                bar_position = x_r + j * bar_width
                ax.bar(bar_position, prob_std_cols[j], width=bar_width, label='k = 2^{}'.format(j + 1))
            ax.set_title('Standard Deviation for different code word lengths k', fontdict=fontdict)
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlabel('q-ary basis q', fontsize=8)
            ax.grid()
            ax.legend()
            ax.set_xticks(x_r + (num_columns - 1) * bar_width / 2)
            ax.set_xticklabels(['$2^{{{}}}$'.format(b) for b in x_r])

            # plt.savefig(save_k)
            # np.save(save_k, prob_std_cols)

            plt.show()
