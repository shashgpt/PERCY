import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns


def getStats(dist):
    m = np.mean(dist)
    std = np.std(dist)
    h = 1.96*std/math.sqrt(len(dist))
    return m, h

def log_dist(distribution):
    distribution = [math.log10( value ) for value in distribution]
    return distribution

def plot_sent_acc_bar_plots(configurations, distributions, FIZ_SIZE):    
    fig_acc, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=FIZ_SIZE)
    plt.grid()
    width = 0.6

    static_no_dist_distributions = []
    static_no_dist_labels = []
    fine_tuning_no_dist_distributions = []
    fine_tuning_no_dist_labels = []
    static_dist_distributions = []
    static_dist_labels = []
    fine_tuning_dist_distributions = []
    fine_tuning_dist_labels = []

    for config in configurations:
        try:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            dataset = config.split("-")[3]
            validation_method = config.split("-")[4]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[5]

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]["STATIC"][dataset][validation_method]["sent_acc"]
                    static_no_dist_distributions.append(distribution)
                    static_no_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['NON_STATIC'][dataset][validation_method]["sent_acc"]
                    fine_tuning_no_dist_distributions.append(distribution)
                    fine_tuning_no_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['STATIC'][dataset][validation_method]['DISTILLATION']["sent_acc"]
                    static_dist_distributions.append(distribution)
                    static_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['NON_STATIC'][dataset][validation_method]['DISTILLATION']["sent_acc"]
                    fine_tuning_dist_distributions.append(distribution)
                    fine_tuning_dist_labels.append(word_embedding)
                except:
                    continue
        except:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            dataset = "SST2"
            validation_method = config.split("-")[3]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[4]

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]["STATIC"][validation_method]["sent_acc"]
                    static_no_dist_distributions.append(distribution)
                    static_no_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['NON_STATIC'][validation_method]["sent_acc"]
                    fine_tuning_no_dist_distributions.append(distribution)
                    fine_tuning_no_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['STATIC'][validation_method]['DISTILLATION']["sent_acc"]
                    static_dist_distributions.append(distribution)
                    static_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['NON_STATIC'][validation_method]['DISTILLATION']["sent_acc"]
                    fine_tuning_dist_distributions.append(distribution)
                    fine_tuning_dist_labels.append(word_embedding)
                except:
                    continue

    static_no_dist_correct_means = [round(getStats(x)[0], 4) for x in static_no_dist_distributions]
    static_no_dist_correct_std = [getStats(x)[1] for x in static_no_dist_distributions]

    fine_tuning_no_dist_correct_means = [round(getStats(x)[0], 4) for x in fine_tuning_no_dist_distributions]
    fine_tuning_no_dist_correct_std = [getStats(x)[1] for x in fine_tuning_no_dist_distributions]

    static_dist_correct_means = [round(getStats(x)[0], 4) for x in static_dist_distributions]
    static_dist_correct_std = [getStats(x)[1] for x in static_dist_distributions]

    fine_tuning_dist_correct_means = [round(getStats(x)[0], 4) for x in fine_tuning_dist_distributions]
    fine_tuning_dist_correct_std = [getStats(x)[1] for x in fine_tuning_dist_distributions]

    static_no_dist_bars = ax1.bar(static_no_dist_labels, static_no_dist_correct_means, width, yerr=static_no_dist_correct_std, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    static_no_dist_error_bars = ax1.errorbar(static_no_dist_labels, static_no_dist_correct_means, yerr=static_no_dist_correct_std, fmt='o', marker='.', capsize=3, capthick=1, markersize=3, elinewidth=0, color='black')
    for rect in static_no_dist_bars:
        height = rect.get_height()
        ax1.text(x=rect.get_x() + rect.get_width() / 2, y=height+0.02, s="{}".format(height), ha='center', fontsize='x-large')
        
    fine_tuning_no_dist_bars = ax2.bar(fine_tuning_no_dist_labels, fine_tuning_no_dist_correct_means, width, yerr=fine_tuning_no_dist_correct_std, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    fine_tuning_no_dist_error_bars = ax2.errorbar(fine_tuning_no_dist_labels, fine_tuning_no_dist_correct_means, yerr=fine_tuning_no_dist_correct_std, fmt='o', marker='.', capsize=3, capthick=1, markersize=3, elinewidth=0, color='black')
    for rect in fine_tuning_no_dist_bars:
        height = rect.get_height()
        ax2.text(x=rect.get_x() + rect.get_width() / 2, y=height+0.02, s="{}".format(height), ha='center', fontsize='x-large')

    static_dist_bars = ax3.bar(static_dist_labels, static_dist_correct_means, width, yerr=static_dist_correct_std, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    static_dist_error_bars = ax3.errorbar(static_dist_labels, static_dist_correct_means, yerr=static_dist_correct_std, fmt='o', marker='.', capsize=3, capthick=1, markersize=3, elinewidth=0, color='black')
    for rect in static_dist_bars:
        height = rect.get_height()
        ax3.text(x=rect.get_x() + rect.get_width() / 2, y=height+0.02, s="{}".format(height), ha='center', fontsize='x-large')

    static_dist_bars = ax4.bar(fine_tuning_dist_labels, fine_tuning_dist_correct_means, width, yerr=fine_tuning_dist_correct_std, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    static_dist_error_bars = ax4.errorbar(fine_tuning_dist_labels, fine_tuning_dist_correct_means, yerr=fine_tuning_dist_correct_std, fmt='o', marker='.', capsize=3, capthick=1, markersize=3, elinewidth=0, color='black')
    for rect in static_dist_bars:
        height = rect.get_height()
        ax4.text(x=rect.get_x() + rect.get_width() / 2, y=height+0.02, s="{}".format(height), ha='center', fontsize='x-large')
        
    # ax.set_ylabel('Rate %')
    ax1.set_title('Static with no distillation('+str(len(static_no_dist_distributions[0]))+')')
    ax2.set_title('Fine-tuning with no distillation('+str(len(fine_tuning_no_dist_distributions[0]))+')')
    ax3.set_title('Static with distillation('+str(len(static_dist_distributions[0]))+')')
    ax4.set_title('Fine-tuning with distillation('+str(len(fine_tuning_dist_distributions[0]))+')')
    params = {'legend.fontsize': 15,
            'axes.labelsize': 25,
            'axes.titlesize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 14,
            'axes.titlepad': 12,
            'axes.axisbelow': True}
    # plt.rc('axes', axisbelow=True)
    fig_acc.suptitle("Sentiment Accuracy", weight="bold")
    plt.yticks([.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    ax1.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    ax2.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    ax3.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    ax4.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    plt.rcParams.update(params)
    ax1.set_ylim(0.0, 1)
    ax2.set_ylim(0.0, 1)
    ax3.set_ylim(0.0, 1)
    ax4.set_ylim(0.0, 1)
    plt.savefig('analysis/CompLing_results/'+dataset+"_"+base_model+'_accuracy.eps', bbox_inches = 'tight')
    fig_acc.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.8, hspace=None)
    plt.show()

def plot_lime_acc_bar_plots(configurations, distributions, FIZ_SIZE):
    fig_acc, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=FIZ_SIZE)
    plt.grid()
    width = 0.6
    static_no_dist_distributions = []
    static_no_dist_labels = []
    fine_tuning_no_dist_distributions = []
    fine_tuning_no_dist_labels = []
    static_dist_distributions = []
    static_dist_labels = []
    fine_tuning_dist_distributions = []
    fine_tuning_dist_labels = []

    for config in configurations:

        try:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            dataset = config.split("-")[3]
            validation_method = config.split("-")[4]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[5]

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]["STATIC"][dataset][validation_method]["ea_values_lime"]
                    static_no_dist_distributions.append(distribution)
                    static_no_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['NON_STATIC'][dataset][validation_method]["ea_values_lime"]
                    fine_tuning_no_dist_distributions.append(distribution)
                    fine_tuning_no_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['STATIC'][dataset][validation_method]['DISTILLATION']["ea_values_lime"]
                    static_dist_distributions.append(distribution)
                    static_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['NON_STATIC'][dataset][validation_method]['DISTILLATION']["ea_values_lime"]
                    fine_tuning_dist_distributions.append(distribution)
                    fine_tuning_dist_labels.append(word_embedding)
                except:
                    continue
        except:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            validation_method = config.split("-")[3]
            dataset = "SST2"
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[4]

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]["STATIC"][validation_method]["ea_values_lime"]
                    static_no_dist_distributions.append(distribution)
                    static_no_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['NON_STATIC'][validation_method]["ea_values_lime"]
                    fine_tuning_no_dist_distributions.append(distribution)
                    fine_tuning_no_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['STATIC'][validation_method]['DISTILLATION']["ea_values_lime"]
                    static_dist_distributions.append(distribution)
                    static_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['NON_STATIC'][validation_method]['DISTILLATION']["ea_values_lime"]
                    fine_tuning_dist_distributions.append(distribution)
                    fine_tuning_dist_labels.append(word_embedding)
                except:
                    continue

    static_no_dist_correct_means = [round(getStats(x)[0], 4) for x in static_no_dist_distributions]
    static_no_dist_correct_std = [getStats(x)[1] for x in static_no_dist_distributions]

    fine_tuning_no_dist_correct_means = [round(getStats(x)[0], 4) for x in fine_tuning_no_dist_distributions]
    fine_tuning_no_dist_correct_std = [getStats(x)[1] for x in fine_tuning_no_dist_distributions]

    static_dist_correct_means = [round(getStats(x)[0], 4) for x in static_dist_distributions]
    static_dist_correct_std = [getStats(x)[1] for x in static_dist_distributions]

    fine_tuning_dist_correct_means = [round(getStats(x)[0], 4) for x in fine_tuning_dist_distributions]
    fine_tuning_dist_correct_std = [getStats(x)[1] for x in fine_tuning_dist_distributions]

    static_no_dist_bars = ax1.bar(static_no_dist_labels, static_no_dist_correct_means, width, yerr=static_no_dist_correct_std, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    static_no_dist_error_bars = ax1.errorbar(static_no_dist_labels, static_no_dist_correct_means, yerr=static_no_dist_correct_std, fmt='o', marker='.', capsize=3, capthick=1, markersize=3, elinewidth=0, color='black')
    for rect in static_no_dist_bars:
        height = rect.get_height()
        ax1.text(x=rect.get_x() + rect.get_width() / 2, y=height+0.02, s="{}".format(height), ha='center', fontsize='x-large')
        
    fine_tuning_no_dist_bars = ax2.bar(fine_tuning_no_dist_labels, fine_tuning_no_dist_correct_means, width, yerr=fine_tuning_no_dist_correct_std, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    fine_tuning_no_dist_error_bars = ax2.errorbar(fine_tuning_no_dist_labels, fine_tuning_no_dist_correct_means, yerr=fine_tuning_no_dist_correct_std, fmt='o', marker='.', capsize=3, capthick=1, markersize=3, elinewidth=0, color='black')
    for rect in fine_tuning_no_dist_bars:
        height = rect.get_height()
        ax2.text(x=rect.get_x() + rect.get_width() / 2, y=height+0.02, s="{}".format(height), ha='center', fontsize='x-large')

    static_dist_bars = ax3.bar(static_dist_labels, static_dist_correct_means, width, yerr=static_dist_correct_std, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    static_dist_error_bars = ax3.errorbar(static_dist_labels, static_dist_correct_means, yerr=static_dist_correct_std, fmt='o', marker='.', capsize=3, capthick=1, markersize=3, elinewidth=0, color='black')
    for rect in static_dist_bars:
        height = rect.get_height()
        ax3.text(x=rect.get_x() + rect.get_width() / 2, y=height+0.02, s="{}".format(height), ha='center', fontsize='x-large')

    static_dist_bars = ax4.bar(fine_tuning_dist_labels, fine_tuning_dist_correct_means, width, yerr=fine_tuning_dist_correct_std, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    static_dist_error_bars = ax4.errorbar(fine_tuning_dist_labels, fine_tuning_dist_correct_means, yerr=fine_tuning_dist_correct_std, fmt='o', marker='.', capsize=3, capthick=1, markersize=3, elinewidth=0, color='black')
    for rect in static_dist_bars:
        height = rect.get_height()
        ax4.text(x=rect.get_x() + rect.get_width() / 2, y=height+0.02, s="{}".format(height), ha='center', fontsize='x-large')
        
    # ax.set_ylabel('Rate %')
    ax1.set_title('Static with no distillation('+str(len(static_no_dist_distributions[0]))+')')
    ax2.set_title('Fine-tuning with no distillation('+str(len(fine_tuning_no_dist_distributions[0]))+')')
    ax3.set_title('Static with distillation('+str(len(static_dist_distributions[0]))+')')
    ax4.set_title('Fine-tuning with distillation('+str(len(fine_tuning_dist_distributions[0]))+')')
    params = {'legend.fontsize': 15,
            'axes.labelsize': 25,
            'axes.titlesize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 14,
            'axes.titlepad': 12,
            'axes.axisbelow': True}
    # plt.rc('axes', axisbelow=True)
    fig_acc.suptitle("LIME explanation acc.", weight="bold")
    plt.yticks([.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    ax1.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    ax2.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    ax3.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    ax4.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    plt.rcParams.update(params)
    ax1.set_ylim(0.0, 1)
    ax2.set_ylim(0.0, 1)
    ax3.set_ylim(0.0, 1)
    ax4.set_ylim(0.0, 1)
    plt.savefig('analysis/CompLing_results/'+dataset+"_"+base_model+'_lime.eps', bbox_inches = 'tight')
    fig_acc.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.8, hspace=None)
    plt.show()

def plot_shap_acc_bar_plots(configurations, distributions, FIZ_SIZE):
    fig_acc, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=FIZ_SIZE)
    plt.grid()
    width = 0.6
    static_no_dist_distributions = []
    static_no_dist_labels = []
    fine_tuning_no_dist_distributions = []
    fine_tuning_no_dist_labels = []
    static_dist_distributions = []
    static_dist_labels = []
    fine_tuning_dist_distributions = []
    fine_tuning_dist_labels = []

    for config in configurations:

        try:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            dataset = config.split("-")[3]
            validation_method = config.split("-")[4]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[5]

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]["STATIC"][dataset][validation_method]["ea_values_shap"]
                    static_no_dist_distributions.append(distribution)
                    static_no_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['NON_STATIC'][dataset][validation_method]["ea_values_shap"]
                    fine_tuning_no_dist_distributions.append(distribution)
                    fine_tuning_no_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['STATIC'][dataset][validation_method]['DISTILLATION']["ea_values_shap"]
                    static_dist_distributions.append(distribution)
                    static_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['NON_STATIC'][dataset][validation_method]['DISTILLATION']["ea_values_shap"]
                    fine_tuning_dist_distributions.append(distribution)
                    fine_tuning_dist_labels.append(word_embedding)
                except:
                    continue
        except:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            validation_method = config.split("-")[3]
            dataset = "SST2"
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[4]

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]["STATIC"][validation_method]["ea_values_shap"]
                    static_no_dist_distributions.append(distribution)
                    static_no_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['NON_STATIC'][validation_method]["ea_values_shap"]
                    fine_tuning_no_dist_distributions.append(distribution)
                    fine_tuning_no_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['STATIC'][validation_method]['DISTILLATION']["ea_values_shap"]
                    static_dist_distributions.append(distribution)
                    static_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['NON_STATIC'][validation_method]['DISTILLATION']["ea_values_shap"]
                    fine_tuning_dist_distributions.append(distribution)
                    fine_tuning_dist_labels.append(word_embedding)
                except:
                    continue

    static_no_dist_correct_means = [round(getStats(x)[0], 4) for x in static_no_dist_distributions]
    static_no_dist_correct_std = [getStats(x)[1] for x in static_no_dist_distributions]

    fine_tuning_no_dist_correct_means = [round(getStats(x)[0], 4) for x in fine_tuning_no_dist_distributions]
    fine_tuning_no_dist_correct_std = [getStats(x)[1] for x in fine_tuning_no_dist_distributions]

    static_dist_correct_means = [round(getStats(x)[0], 4) for x in static_dist_distributions]
    static_dist_correct_std = [getStats(x)[1] for x in static_dist_distributions]

    fine_tuning_dist_correct_means = [round(getStats(x)[0], 4) for x in fine_tuning_dist_distributions]
    fine_tuning_dist_correct_std = [getStats(x)[1] for x in fine_tuning_dist_distributions]

    static_no_dist_bars = ax1.bar(static_no_dist_labels, static_no_dist_correct_means, width, yerr=static_no_dist_correct_std, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    static_no_dist_error_bars = ax1.errorbar(static_no_dist_labels, static_no_dist_correct_means, yerr=static_no_dist_correct_std, fmt='o', marker='.', capsize=3, capthick=1, markersize=3, elinewidth=0, color='black')
    for rect in static_no_dist_bars:
        height = rect.get_height()
        ax1.text(x=rect.get_x() + rect.get_width() / 2, y=height+0.02, s="{}".format(height), ha='center', fontsize='x-large')
        
    fine_tuning_no_dist_bars = ax2.bar(fine_tuning_no_dist_labels, fine_tuning_no_dist_correct_means, width, yerr=fine_tuning_no_dist_correct_std, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    fine_tuning_no_dist_error_bars = ax2.errorbar(fine_tuning_no_dist_labels, fine_tuning_no_dist_correct_means, yerr=fine_tuning_no_dist_correct_std, fmt='o', marker='.', capsize=3, capthick=1, markersize=3, elinewidth=0, color='black')
    for rect in fine_tuning_no_dist_bars:
        height = rect.get_height()
        ax2.text(x=rect.get_x() + rect.get_width() / 2, y=height+0.02, s="{}".format(height), ha='center', fontsize='x-large')

    static_dist_bars = ax3.bar(static_dist_labels, static_dist_correct_means, width, yerr=static_dist_correct_std, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    static_dist_error_bars = ax3.errorbar(static_dist_labels, static_dist_correct_means, yerr=static_dist_correct_std, fmt='o', marker='.', capsize=3, capthick=1, markersize=3, elinewidth=0, color='black')
    for rect in static_dist_bars:
        height = rect.get_height()
        ax3.text(x=rect.get_x() + rect.get_width() / 2, y=height+0.02, s="{}".format(height), ha='center', fontsize='x-large')

    static_dist_bars = ax4.bar(fine_tuning_dist_labels, fine_tuning_dist_correct_means, width, yerr=fine_tuning_dist_correct_std, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    static_dist_error_bars = ax4.errorbar(fine_tuning_dist_labels, fine_tuning_dist_correct_means, yerr=fine_tuning_dist_correct_std, fmt='o', marker='.', capsize=3, capthick=1, markersize=3, elinewidth=0, color='black')
    for rect in static_dist_bars:
        height = rect.get_height()
        ax4.text(x=rect.get_x() + rect.get_width() / 2, y=height+0.02, s="{}".format(height), ha='center', fontsize='x-large')
        
    # ax.set_ylabel('Rate %')
    ax1.set_title('Static with no distillation('+str(len(static_no_dist_distributions[0]))+')')
    ax2.set_title('Fine-tuning with no distillation('+str(len(fine_tuning_no_dist_distributions[0]))+')')
    ax3.set_title('Static with distillation('+str(len(static_dist_distributions[0]))+')')
    ax4.set_title('Fine-tuning with distillation('+str(len(fine_tuning_dist_distributions[0]))+')')
    params = {'legend.fontsize': 15,
            'axes.labelsize': 25,
            'axes.titlesize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 14,
            'axes.titlepad': 12,
            'axes.axisbelow': True}
    # plt.rc('axes', axisbelow=True)
    fig_acc.suptitle("SHAP explanation acc.", weight="bold")
    plt.yticks([.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    ax1.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    ax2.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    ax3.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    ax4.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    plt.rcParams.update(params)
    ax1.set_ylim(0.0, 1)
    ax2.set_ylim(0.0, 1)
    ax3.set_ylim(0.0, 1)
    ax4.set_ylim(0.0, 1)
    plt.savefig('analysis/CompLing_results/'+dataset+"_"+base_model+'_shap.eps', bbox_inches = 'tight')
    fig_acc.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.8, hspace=None)
    plt.show()

def plot_int_grad_acc_bar_plots(configurations, distributions, FIZ_SIZE):
    fig_acc, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=FIZ_SIZE)
    plt.grid()
    width = 0.6
    static_no_dist_distributions = []
    static_no_dist_labels = []
    fine_tuning_no_dist_distributions = []
    fine_tuning_no_dist_labels = []
    static_dist_distributions = []
    static_dist_labels = []
    fine_tuning_dist_distributions = []
    fine_tuning_dist_labels = []

    for config in configurations:

        try:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            dataset = config.split("-")[3]
            validation_method = config.split("-")[4]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[5]

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]["STATIC"][dataset][validation_method]["ea_values_int_grad"]
                    static_no_dist_distributions.append(distribution)
                    static_no_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['NON_STATIC'][dataset][validation_method]["ea_values_int_grad"]
                    fine_tuning_no_dist_distributions.append(distribution)
                    fine_tuning_no_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['STATIC'][dataset][validation_method]['DISTILLATION']["ea_values_int_grad"]
                    static_dist_distributions.append(distribution)
                    static_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['NON_STATIC'][dataset][validation_method]['DISTILLATION']["ea_values_int_grad"]
                    fine_tuning_dist_distributions.append(distribution)
                    fine_tuning_dist_labels.append(word_embedding)
                except:
                    continue
        except:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            validation_method = config.split("-")[3]
            dataset = "SST2"
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[4]

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]["STATIC"][validation_method]["ea_values_int_grad"]
                    static_no_dist_distributions.append(distribution)
                    static_no_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['NON_STATIC'][validation_method]["ea_values_int_grad"]
                    fine_tuning_no_dist_distributions.append(distribution)
                    fine_tuning_no_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['STATIC'][validation_method]['DISTILLATION']["ea_values_int_grad"]
                    static_dist_distributions.append(distribution)
                    static_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['NON_STATIC'][validation_method]['DISTILLATION']["ea_values_int_grad"]
                    fine_tuning_dist_distributions.append(distribution)
                    fine_tuning_dist_labels.append(word_embedding)
                except:
                    continue

    static_no_dist_correct_means = [round(getStats(x)[0], 4) for x in static_no_dist_distributions]
    static_no_dist_correct_std = [getStats(x)[1] for x in static_no_dist_distributions]

    fine_tuning_no_dist_correct_means = [round(getStats(x)[0], 4) for x in fine_tuning_no_dist_distributions]
    fine_tuning_no_dist_correct_std = [getStats(x)[1] for x in fine_tuning_no_dist_distributions]

    static_dist_correct_means = [round(getStats(x)[0], 4) for x in static_dist_distributions]
    static_dist_correct_std = [getStats(x)[1] for x in static_dist_distributions]

    fine_tuning_dist_correct_means = [round(getStats(x)[0], 4) for x in fine_tuning_dist_distributions]
    fine_tuning_dist_correct_std = [getStats(x)[1] for x in fine_tuning_dist_distributions]

    static_no_dist_bars = ax1.bar(static_no_dist_labels, static_no_dist_correct_means, width, yerr=static_no_dist_correct_std, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    static_no_dist_error_bars = ax1.errorbar(static_no_dist_labels, static_no_dist_correct_means, yerr=static_no_dist_correct_std, fmt='o', marker='.', capsize=3, capthick=1, markersize=3, elinewidth=0, color='black')
    for rect in static_no_dist_bars:
        height = rect.get_height()
        ax1.text(x=rect.get_x() + rect.get_width() / 2, y=height+0.02, s="{}".format(height), ha='center', fontsize='x-large')
        
    fine_tuning_no_dist_bars = ax2.bar(fine_tuning_no_dist_labels, fine_tuning_no_dist_correct_means, width, yerr=fine_tuning_no_dist_correct_std, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    fine_tuning_no_dist_error_bars = ax2.errorbar(fine_tuning_no_dist_labels, fine_tuning_no_dist_correct_means, yerr=fine_tuning_no_dist_correct_std, fmt='o', marker='.', capsize=3, capthick=1, markersize=3, elinewidth=0, color='black')
    for rect in fine_tuning_no_dist_bars:
        height = rect.get_height()
        ax2.text(x=rect.get_x() + rect.get_width() / 2, y=height+0.02, s="{}".format(height), ha='center', fontsize='x-large')

    static_dist_bars = ax3.bar(static_dist_labels, static_dist_correct_means, width, yerr=static_dist_correct_std, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    static_dist_error_bars = ax3.errorbar(static_dist_labels, static_dist_correct_means, yerr=static_dist_correct_std, fmt='o', marker='.', capsize=3, capthick=1, markersize=3, elinewidth=0, color='black')
    for rect in static_dist_bars:
        height = rect.get_height()
        ax3.text(x=rect.get_x() + rect.get_width() / 2, y=height+0.02, s="{}".format(height), ha='center', fontsize='x-large')

    static_dist_bars = ax4.bar(fine_tuning_dist_labels, fine_tuning_dist_correct_means, width, yerr=fine_tuning_dist_correct_std, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    static_dist_error_bars = ax4.errorbar(fine_tuning_dist_labels, fine_tuning_dist_correct_means, yerr=fine_tuning_dist_correct_std, fmt='o', marker='.', capsize=3, capthick=1, markersize=3, elinewidth=0, color='black')
    for rect in static_dist_bars:
        height = rect.get_height()
        ax4.text(x=rect.get_x() + rect.get_width() / 2, y=height+0.02, s="{}".format(height), ha='center', fontsize='x-large')
        
    # ax.set_ylabel('Rate %')
    ax1.set_title('Static with no distillation('+str(len(static_no_dist_distributions[0]))+')')
    ax2.set_title('Fine-tuning with no distillation('+str(len(fine_tuning_no_dist_distributions[0]))+')')
    ax3.set_title('Static with distillation('+str(len(static_dist_distributions[0]))+')')
    ax4.set_title('Fine-tuning with distillation('+str(len(fine_tuning_dist_distributions[0]))+')')
    params = {'legend.fontsize': 15,
            'axes.labelsize': 25,
            'axes.titlesize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 14,
            'axes.titlepad': 12,
            'axes.axisbelow': True}
    # plt.rc('axes', axisbelow=True)
    fig_acc.suptitle("INT. GRAD. explanation acc.", weight="bold")
    plt.yticks([.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    ax1.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    ax2.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    ax3.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    ax4.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    plt.rcParams.update(params)
    ax1.set_ylim(0.0, 1)
    ax2.set_ylim(0.0, 1)
    ax3.set_ylim(0.0, 1)
    ax4.set_ylim(0.0, 1)
    plt.savefig('analysis/CompLing_results/'+dataset+"_"+base_model+'_int_grad.eps', bbox_inches = 'tight')
    fig_acc.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.8, hspace=None)
    plt.show()

def plot_lipschitz_box_plots(configurations, distributions, FIZ_SIZE, y_limits):

    fig_acc, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=FIZ_SIZE)
    plt.grid()
    width = 0.6

    lime_lip_values = []
    shap_lip_values = []
    int_grad_lip_values = []

    labels = ["LIME", "SHAP", "INT-GRAD"]

    for config in configurations:

        try:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            dataset = config.split("-")[3]
            validation_method = config.split("-")[4]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[5]

            # LIME values
            try:
                lime_lip_values_dist = distributions[base_model][word_vectors][fine_tuning][dataset][validation_method]["lime_lipschitz_values"]
                for lip_val in lime_lip_values_dist:
                    lime_lip_values.append(lip_val)
            except:
                lime_lip_values_dist = distributions[base_model][word_vectors][fine_tuning][dataset][validation_method][distillation]["lime_lipschitz_values"]
                for lip_val in lime_lip_values_dist:
                    lime_lip_values.append(lip_val)
            
            # SHAP values
            try:
                shap_lip_values_dist = distributions[base_model][word_vectors][fine_tuning][dataset][validation_method]["shap_lipschitz_values"]
                for lip_val in shap_lip_values_dist:
                    shap_lip_values.append(lip_val)
            except:
                shap_lip_values_dist = distributions[base_model][word_vectors][fine_tuning][dataset][validation_method][distillation]["shap_lipschitz_values"]
                for lip_val in shap_lip_values_dist:
                    shap_lip_values.append(lip_val)
            
            # INT-GRAD values
            try:
                int_grad_lip_values_dist = distributions[base_model][word_vectors][fine_tuning][dataset][validation_method]["int_grad_lipschitz_values"]
                for lip_val in int_grad_lip_values_dist:
                    int_grad_lip_values.append(lip_val)
            except:
                int_grad_lip_values_dist = distributions[base_model][word_vectors][fine_tuning][dataset][validation_method][distillation]["int_grad_lipschitz_values"]
                for lip_val in int_grad_lip_values_dist:
                    int_grad_lip_values.append(lip_val)
        
        except:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            dataset = "SST2"
            validation_method = config.split("-")[3]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[4]

            # LIME values
            try:
                lime_lip_values_dist = distributions[base_model][word_vectors][fine_tuning][validation_method]["lime_lipschitz_values"]
                for lip_val in lime_lip_values_dist:
                    lime_lip_values.append(lip_val)
            except:
                lime_lip_values_dist = distributions[base_model][word_vectors][fine_tuning][validation_method][distillation]["lime_lipschitz_values"]
                for lip_val in lime_lip_values_dist:
                    lime_lip_values.append(lip_val)
            
            # SHAP values
            try:
                shap_lip_values_dist = distributions[base_model][word_vectors][fine_tuning][validation_method]["shap_lipschitz_values"]
                for lip_val in shap_lip_values_dist:
                    shap_lip_values.append(lip_val)
            except:
                shap_lip_values_dist = distributions[base_model][word_vectors][fine_tuning][validation_method][distillation]["shap_lipschitz_values"]
                for lip_val in shap_lip_values_dist:
                    shap_lip_values.append(lip_val)
            
            # INT-GRAD values
            try:
                int_grad_lip_values_dist = distributions[base_model][word_vectors][fine_tuning][validation_method]["int_grad_lipschitz_values"]
                for lip_val in int_grad_lip_values_dist:
                    int_grad_lip_values.append(lip_val)
            except:
                int_grad_lip_values_dist = distributions[base_model][word_vectors][fine_tuning][validation_method][distillation]["int_grad_lipschitz_values"]
                for lip_val in int_grad_lip_values_dist:
                    int_grad_lip_values.append(lip_val)

    sns.boxplot(data=lime_lip_values, palette=["red", "lightblue"], ax=ax1)
    sns.boxplot(data=shap_lip_values, palette=["red", "lightblue"], ax=ax2)
    sns.boxplot(data=int_grad_lip_values, palette=["red", "lightblue"], ax=ax3)
        
    # ax.set_ylabel('Rate %')
    # ax.set_title('Lipschitz scores on SENTIMENT140 dataset('+str(len(lime_lip_values[0]))+')')

    ax1.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    ax2.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    ax3.grid(color = 'black', linestyle = '--', linewidth = 0.5)

    ax1.set_xticklabels(["LIME"], weight="roman")
    ax2.set_xticklabels(["SHAP"], weight="roman")
    ax3.set_xticklabels(["INT-GRAD"], weight="roman")

    ax1.set_ylim(y_limits[0], y_limits[1])
    ax2.set_ylim(y_limits[0], y_limits[1])
    ax3.set_ylim(y_limits[0], y_limits[1])

    params = {'legend.fontsize': 15,
            'axes.labelsize': 25,
            'axes.titlesize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 14,
            'axes.titlepad': 12,
            'axes.axisbelow': True}

    # plt.rc('axes', axisbelow=True)
    fig_acc.suptitle("Lipschitz scores on "+dataset+" dataset", weight="bold")

    plt.rcParams.update(params)
    plt.savefig('analysis/CompLing_results/lip.eps', bbox_inches = 'tight')
    fig_acc.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.8, hspace=None)
    plt.show()

def plot_lime_shap_consistency_bar_plots(configurations, distributions, FIZ_SIZE):

    fig_acc, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=FIZ_SIZE)
    plt.grid()

    width = 0.6

    static_no_dist_distributions = []
    static_no_dist_labels = []
    fine_tuning_no_dist_distributions = []
    fine_tuning_no_dist_labels = []
    static_dist_distributions = []
    static_dist_labels = []
    fine_tuning_dist_distributions = []
    fine_tuning_dist_labels = []
    
    for config in configurations:

        try:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            dataset = config.split("-")[3]
            validation_method = config.split("-")[4]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[5]

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]["STATIC"][dataset][validation_method]["lime_shap_consistency_values"]
                    static_no_dist_distributions.append(distribution)
                    static_no_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['NON_STATIC'][dataset][validation_method]["lime_shap_consistency_values"]
                    fine_tuning_no_dist_distributions.append(distribution)
                    fine_tuning_no_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['STATIC'][dataset][validation_method]['DISTILLATION']["lime_shap_consistency_values"]
                    static_dist_distributions.append(distribution)
                    static_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['NON_STATIC'][dataset][validation_method]['DISTILLATION']["lime_shap_consistency_values"]
                    fine_tuning_dist_distributions.append(distribution)
                    fine_tuning_dist_labels.append(word_embedding)
                except:
                    continue
        except:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            dataset = "SST2"
            validation_method = config.split("-")[3]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[4]

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]["STATIC"][validation_method]["lime_shap_consistency_values"]
                    static_no_dist_distributions.append(distribution)
                    static_no_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['NON_STATIC'][validation_method]["lime_shap_consistency_values"]
                    fine_tuning_no_dist_distributions.append(distribution)
                    fine_tuning_no_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['STATIC'][validation_method]['DISTILLATION']["lime_shap_consistency_values"]
                    static_dist_distributions.append(distribution)
                    static_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['NON_STATIC'][validation_method]['DISTILLATION']["lime_shap_consistency_values"]
                    fine_tuning_dist_distributions.append(distribution)
                    fine_tuning_dist_labels.append(word_embedding)
                except:
                    continue
    
    static_no_dist_correct_means = [round(getStats(x)[0], 4) for x in static_no_dist_distributions]
    static_no_dist_correct_std = [getStats(x)[1] for x in static_no_dist_distributions]

    fine_tuning_no_dist_correct_means = [round(getStats(x)[0], 4) for x in fine_tuning_no_dist_distributions]
    fine_tuning_no_dist_correct_std = [getStats(x)[1] for x in fine_tuning_no_dist_distributions]

    static_dist_correct_means = [round(getStats(x)[0], 4) for x in static_dist_distributions]
    static_dist_correct_std = [getStats(x)[1] for x in static_dist_distributions]

    fine_tuning_dist_correct_means = [round(getStats(x)[0], 4) for x in fine_tuning_dist_distributions]
    fine_tuning_dist_correct_std = [getStats(x)[1] for x in fine_tuning_dist_distributions]

    static_no_dist_bars = ax1.bar(static_no_dist_labels, static_no_dist_correct_means, width, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    # static_no_dist_bars = ax1.bar(static_no_dist_labels, static_no_dist_correct_means, width, yerr=static_no_dist_correct_std, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    # static_no_dist_error_bars = ax1.errorbar(static_no_dist_labels, static_no_dist_correct_means, yerr=static_no_dist_correct_std, fmt='o', marker='.', capsize=3, capthick=1, markersize=3, elinewidth=0, color='black')
    for rect in static_no_dist_bars:
        height = rect.get_height()
        ax1.text(x=rect.get_x() + rect.get_width() / 2, y=height+0.02, s="{}".format(height), ha='center', fontsize='x-large')
    
    fine_tuning_no_dist_bars = ax2.bar(fine_tuning_no_dist_labels, fine_tuning_no_dist_correct_means, width, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    # fine_tuning_no_dist_bars = ax2.bar(fine_tuning_no_dist_labels, fine_tuning_no_dist_correct_means, width, yerr=fine_tuning_no_dist_correct_std, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    # fine_tuning_no_dist_error_bars = ax2.errorbar(fine_tuning_no_dist_labels, fine_tuning_no_dist_correct_means, yerr=fine_tuning_no_dist_correct_std, fmt='o', marker='.', capsize=3, capthick=1, markersize=3, elinewidth=0, color='black')
    for rect in fine_tuning_no_dist_bars:
        height = rect.get_height()
        ax2.text(x=rect.get_x() + rect.get_width() / 2, y=height+0.02, s="{}".format(height), ha='center', fontsize='x-large')

    static_dist_bars = ax3.bar(static_dist_labels, static_dist_correct_means, width, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    # static_dist_bars = ax3.bar(static_dist_labels, static_dist_correct_means, width, yerr=static_dist_correct_std, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    # static_dist_error_bars = ax3.errorbar(static_dist_labels, static_dist_correct_means, yerr=static_dist_correct_std, fmt='o', marker='.', capsize=3, capthick=1, markersize=3, elinewidth=0, color='black')
    for rect in static_dist_bars:
        height = rect.get_height()
        ax3.text(x=rect.get_x() + rect.get_width() / 2, y=height+0.02, s="{}".format(height), ha='center', fontsize='x-large')

    static_dist_bars = ax4.bar(fine_tuning_dist_labels, fine_tuning_dist_correct_means, width, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    # static_dist_bars = ax4.bar(fine_tuning_dist_labels, fine_tuning_dist_correct_means, width, yerr=fine_tuning_dist_correct_std, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    # static_dist_error_bars = ax4.errorbar(fine_tuning_dist_labels, fine_tuning_dist_correct_means, yerr=fine_tuning_dist_correct_std, fmt='o', marker='.', capsize=3, capthick=1, markersize=3, elinewidth=0, color='black')
    for rect in static_dist_bars:
        height = rect.get_height()
        ax4.text(x=rect.get_x() + rect.get_width() / 2, y=height+0.02, s="{}".format(height), ha='center', fontsize='x-large')
        
    # ax.set_ylabel('Rate %')
    ax1.set_title('Static with no distillation('+str(len(static_no_dist_distributions[0]))+')')
    ax2.set_title('Fine-tuning with no distillation('+str(len(fine_tuning_no_dist_distributions[0]))+')')
    ax3.set_title('Static with distillation('+str(len(static_dist_distributions[0]))+')')
    ax4.set_title('Fine-tuning with distillation('+str(len(fine_tuning_dist_distributions[0]))+')')
    params = {'legend.fontsize': 15,
            'axes.labelsize': 25,
            'axes.titlesize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 14,
            'axes.titlepad': 12,
            'axes.axisbelow': True}
    # plt.rc('axes', axisbelow=True)
    fig_acc.suptitle("LIME. SHAP. consistency", weight="bold")
    plt.yticks([.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    ax1.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    ax2.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    ax3.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    ax4.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    plt.rcParams.update(params)
    ax1.set_ylim(0.0, 1)
    ax2.set_ylim(0.0, 1)
    ax3.set_ylim(0.0, 1)
    ax4.set_ylim(0.0, 1)
    # plt.savefig('analysis/CompLing_results/int_grad.eps', bbox_inches = 'tight')
    fig_acc.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.8, hspace=None)
    plt.show()

def plot_lime_int_grad_consistency_bar_plots(configurations, distributions, FIZ_SIZE):

    fig_acc, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=FIZ_SIZE)
    plt.grid()

    width = 0.6

    static_no_dist_distributions = []
    static_no_dist_labels = []
    fine_tuning_no_dist_distributions = []
    fine_tuning_no_dist_labels = []
    static_dist_distributions = []
    static_dist_labels = []
    fine_tuning_dist_distributions = []
    fine_tuning_dist_labels = []

    for config in configurations:

        try:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            dataset = config.split("-")[3]
            validation_method = config.split("-")[4]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[5]

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]["STATIC"][dataset][validation_method]["lime_int_grad_consistency_values"]
                    static_no_dist_distributions.append(distribution)
                    static_no_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['NON_STATIC'][dataset][validation_method]["lime_int_grad_consistency_values"]
                    fine_tuning_no_dist_distributions.append(distribution)
                    fine_tuning_no_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['STATIC'][dataset][validation_method]['DISTILLATION']["lime_int_grad_consistency_values"]
                    static_dist_distributions.append(distribution)
                    static_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['NON_STATIC'][dataset][validation_method]['DISTILLATION']["lime_int_grad_consistency_values"]
                    fine_tuning_dist_distributions.append(distribution)
                    fine_tuning_dist_labels.append(word_embedding)
                except:
                    continue
        except:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            dataset = "SST2"
            validation_method = config.split("-")[3]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[4]

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]["STATIC"][validation_method]["lime_int_grad_consistency_values"]
                    static_no_dist_distributions.append(distribution)
                    static_no_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['NON_STATIC'][validation_method]["lime_int_grad_consistency_values"]
                    fine_tuning_no_dist_distributions.append(distribution)
                    fine_tuning_no_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['STATIC'][validation_method]['DISTILLATION']["lime_int_grad_consistency_values"]
                    static_dist_distributions.append(distribution)
                    static_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['NON_STATIC'][validation_method]['DISTILLATION']["lime_int_grad_consistency_values"]
                    fine_tuning_dist_distributions.append(distribution)
                    fine_tuning_dist_labels.append(word_embedding)
                except:
                    continue
    
    # lime_int_grad_consistency_values

    static_no_dist_correct_means = [round(getStats(x)[0], 4) for x in static_no_dist_distributions]
    static_no_dist_correct_std = [getStats(x)[1] for x in static_no_dist_distributions]

    fine_tuning_no_dist_correct_means = [round(getStats(x)[0], 4) for x in fine_tuning_no_dist_distributions]
    fine_tuning_no_dist_correct_std = [getStats(x)[1] for x in fine_tuning_no_dist_distributions]

    static_dist_correct_means = [round(getStats(x)[0], 4) for x in static_dist_distributions]
    static_dist_correct_std = [getStats(x)[1] for x in static_dist_distributions]

    fine_tuning_dist_correct_means = [round(getStats(x)[0], 4) for x in fine_tuning_dist_distributions]
    fine_tuning_dist_correct_std = [getStats(x)[1] for x in fine_tuning_dist_distributions]

    static_no_dist_bars = ax1.bar(static_no_dist_labels, static_no_dist_correct_means, width, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    # static_no_dist_bars = ax1.bar(static_no_dist_labels, static_no_dist_correct_means, width, yerr=static_no_dist_correct_std, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    # static_no_dist_error_bars = ax1.errorbar(static_no_dist_labels, static_no_dist_correct_means, yerr=static_no_dist_correct_std, fmt='o', marker='.', capsize=3, capthick=1, markersize=3, elinewidth=0, color='black')
    for rect in static_no_dist_bars:
        height = rect.get_height()
        ax1.text(x=rect.get_x() + rect.get_width() / 2, y=height+0.02, s="{}".format(height), ha='center', fontsize='x-large')
    
    fine_tuning_no_dist_bars = ax2.bar(fine_tuning_no_dist_labels, fine_tuning_no_dist_correct_means, width, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    # fine_tuning_no_dist_bars = ax2.bar(fine_tuning_no_dist_labels, fine_tuning_no_dist_correct_means, width, yerr=fine_tuning_no_dist_correct_std, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    # fine_tuning_no_dist_error_bars = ax2.errorbar(fine_tuning_no_dist_labels, fine_tuning_no_dist_correct_means, yerr=fine_tuning_no_dist_correct_std, fmt='o', marker='.', capsize=3, capthick=1, markersize=3, elinewidth=0, color='black')
    for rect in fine_tuning_no_dist_bars:
        height = rect.get_height()
        ax2.text(x=rect.get_x() + rect.get_width() / 2, y=height+0.02, s="{}".format(height), ha='center', fontsize='x-large')

    static_dist_bars = ax3.bar(static_dist_labels, static_dist_correct_means, width, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    # static_dist_bars = ax3.bar(static_dist_labels, static_dist_correct_means, width, yerr=static_dist_correct_std, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    # static_dist_error_bars = ax3.errorbar(static_dist_labels, static_dist_correct_means, yerr=static_dist_correct_std, fmt='o', marker='.', capsize=3, capthick=1, markersize=3, elinewidth=0, color='black')
    for rect in static_dist_bars:
        height = rect.get_height()
        ax3.text(x=rect.get_x() + rect.get_width() / 2, y=height+0.02, s="{}".format(height), ha='center', fontsize='x-large')

    static_dist_bars = ax4.bar(fine_tuning_dist_labels, fine_tuning_dist_correct_means, width, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    # static_dist_bars = ax4.bar(fine_tuning_dist_labels, fine_tuning_dist_correct_means, width, yerr=fine_tuning_dist_correct_std, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    # static_dist_error_bars = ax4.errorbar(fine_tuning_dist_labels, fine_tuning_dist_correct_means, yerr=fine_tuning_dist_correct_std, fmt='o', marker='.', capsize=3, capthick=1, markersize=3, elinewidth=0, color='black')
    for rect in static_dist_bars:
        height = rect.get_height()
        ax4.text(x=rect.get_x() + rect.get_width() / 2, y=height+0.02, s="{}".format(height), ha='center', fontsize='x-large')
        
    # ax.set_ylabel('Rate %')
    ax1.set_title('Static with no distillation('+str(len(static_no_dist_distributions[0]))+')')
    ax2.set_title('Fine-tuning with no distillation('+str(len(fine_tuning_no_dist_distributions[0]))+')')
    ax3.set_title('Static with distillation('+str(len(static_dist_distributions[0]))+')')
    ax4.set_title('Fine-tuning with distillation('+str(len(fine_tuning_dist_distributions[0]))+')')
    params = {'legend.fontsize': 15,
            'axes.labelsize': 25,
            'axes.titlesize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 14,
            'axes.titlepad': 12,
            'axes.axisbelow': True}
    # plt.rc('axes', axisbelow=True)
    fig_acc.suptitle("LIME. INT. GRAD. consistency", weight="bold")
    plt.yticks([.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    ax1.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    ax2.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    ax3.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    ax4.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    plt.rcParams.update(params)
    ax1.set_ylim(0.0, 1)
    ax2.set_ylim(0.0, 1)
    ax3.set_ylim(0.0, 1)
    ax4.set_ylim(0.0, 1)
    # plt.savefig('analysis/CompLing_results/int_grad.eps', bbox_inches = 'tight')
    fig_acc.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.8, hspace=None)
    plt.show()

def plot_shap_int_grad_consistency_bar_plots(configurations, distributions, FIZ_SIZE):

    fig_acc, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=FIZ_SIZE)
    plt.grid()

    width = 0.6

    static_no_dist_distributions = []
    static_no_dist_labels = []
    fine_tuning_no_dist_distributions = []
    fine_tuning_no_dist_labels = []
    static_dist_distributions = []
    static_dist_labels = []
    fine_tuning_dist_distributions = []
    fine_tuning_dist_labels = []

    for config in configurations:
        try:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            dataset = config.split("-")[3]
            validation_method = config.split("-")[4]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[5]

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]["STATIC"][dataset][validation_method]["shap_int_grad_consistency_values"]
                    static_no_dist_distributions.append(distribution)
                    static_no_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['NON_STATIC'][dataset][validation_method]["shap_int_grad_consistency_values"]
                    fine_tuning_no_dist_distributions.append(distribution)
                    fine_tuning_no_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['STATIC'][dataset][validation_method]['DISTILLATION']["shap_int_grad_consistency_values"]
                    static_dist_distributions.append(distribution)
                    static_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['NON_STATIC'][dataset][validation_method]['DISTILLATION']["shap_int_grad_consistency_values"]
                    fine_tuning_dist_distributions.append(distribution)
                    fine_tuning_dist_labels.append(word_embedding)
                except:
                    continue
        except:
            base_model = config.split("-")[0]
            word_vectors = config.split("-")[1]
            fine_tuning = config.split("-")[2]
            dataset = "SST2"
            validation_method = config.split("-")[3]
            if "DISTILLATION" in list(config.split("-")):
                distillation = config.split("-")[4]

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]["STATIC"][validation_method]["shap_int_grad_consistency_values"]
                    static_no_dist_distributions.append(distribution)
                    static_no_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['NON_STATIC'][validation_method]["shap_int_grad_consistency_values"]
                    fine_tuning_no_dist_distributions.append(distribution)
                    fine_tuning_no_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['STATIC'][validation_method]['DISTILLATION']["shap_int_grad_consistency_values"]
                    static_dist_distributions.append(distribution)
                    static_dist_labels.append(word_embedding)
                except:
                    continue

            for word_embedding in ['WORD2VEC', 'GLOVE', 'ELMO', 'BERT']:
                try:
                    distribution = distributions[base_model][word_embedding]['NON_STATIC'][validation_method]['DISTILLATION']["shap_int_grad_consistency_values"]
                    fine_tuning_dist_distributions.append(distribution)
                    fine_tuning_dist_labels.append(word_embedding)
                except:
                    continue
    
    # shap_int_grad_consistency_values

    static_no_dist_correct_means = [round(getStats(x)[0], 4) for x in static_no_dist_distributions]
    static_no_dist_correct_std = [getStats(x)[1] for x in static_no_dist_distributions]

    fine_tuning_no_dist_correct_means = [round(getStats(x)[0], 4) for x in fine_tuning_no_dist_distributions]
    fine_tuning_no_dist_correct_std = [getStats(x)[1] for x in fine_tuning_no_dist_distributions]

    static_dist_correct_means = [round(getStats(x)[0], 4) for x in static_dist_distributions]
    static_dist_correct_std = [getStats(x)[1] for x in static_dist_distributions]

    fine_tuning_dist_correct_means = [round(getStats(x)[0], 4) for x in fine_tuning_dist_distributions]
    fine_tuning_dist_correct_std = [getStats(x)[1] for x in fine_tuning_dist_distributions]

    static_no_dist_bars = ax1.bar(static_no_dist_labels, static_no_dist_correct_means, width, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    # static_no_dist_bars = ax1.bar(static_no_dist_labels, static_no_dist_correct_means, width, yerr=static_no_dist_correct_std, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    # static_no_dist_error_bars = ax1.errorbar(static_no_dist_labels, static_no_dist_correct_means, yerr=static_no_dist_correct_std, fmt='o', marker='.', capsize=3, capthick=1, markersize=3, elinewidth=0, color='black')
    for rect in static_no_dist_bars:
        height = rect.get_height()
        ax1.text(x=rect.get_x() + rect.get_width() / 2, y=height+0.02, s="{}".format(height), ha='center', fontsize='x-large')
    
    fine_tuning_no_dist_bars = ax2.bar(fine_tuning_no_dist_labels, fine_tuning_no_dist_correct_means, width, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    # fine_tuning_no_dist_bars = ax2.bar(fine_tuning_no_dist_labels, fine_tuning_no_dist_correct_means, width, yerr=fine_tuning_no_dist_correct_std, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    # fine_tuning_no_dist_error_bars = ax2.errorbar(fine_tuning_no_dist_labels, fine_tuning_no_dist_correct_means, yerr=fine_tuning_no_dist_correct_std, fmt='o', marker='.', capsize=3, capthick=1, markersize=3, elinewidth=0, color='black')
    for rect in fine_tuning_no_dist_bars:
        height = rect.get_height()
        ax2.text(x=rect.get_x() + rect.get_width() / 2, y=height+0.02, s="{}".format(height), ha='center', fontsize='x-large')

    static_dist_bars = ax3.bar(static_dist_labels, static_dist_correct_means, width, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    # static_dist_bars = ax3.bar(static_dist_labels, static_dist_correct_means, width, yerr=static_dist_correct_std, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    # static_dist_error_bars = ax3.errorbar(static_dist_labels, static_dist_correct_means, yerr=static_dist_correct_std, fmt='o', marker='.', capsize=3, capthick=1, markersize=3, elinewidth=0, color='black')
    for rect in static_dist_bars:
        height = rect.get_height()
        ax3.text(x=rect.get_x() + rect.get_width() / 2, y=height+0.02, s="{}".format(height), ha='center', fontsize='x-large')

    static_dist_bars = ax4.bar(fine_tuning_dist_labels, fine_tuning_dist_correct_means, width, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    # static_dist_bars = ax4.bar(fine_tuning_dist_labels, fine_tuning_dist_correct_means, width, yerr=fine_tuning_dist_correct_std, edgecolor = 'black', linewidth = 1, color=['#FF3400','#0025FF','#008000','#FFFD07'])
    # static_dist_error_bars = ax4.errorbar(fine_tuning_dist_labels, fine_tuning_dist_correct_means, yerr=fine_tuning_dist_correct_std, fmt='o', marker='.', capsize=3, capthick=1, markersize=3, elinewidth=0, color='black')
    for rect in static_dist_bars:
        height = rect.get_height()
        ax4.text(x=rect.get_x() + rect.get_width() / 2, y=height+0.02, s="{}".format(height), ha='center', fontsize='x-large')
        
    # ax.set_ylabel('Rate %')
    ax1.set_title('Static with no distillation('+str(len(static_no_dist_distributions[0]))+')')
    ax2.set_title('Fine-tuning with no distillation('+str(len(fine_tuning_no_dist_distributions[0]))+')')
    ax3.set_title('Static with distillation('+str(len(static_dist_distributions[0]))+')')
    ax4.set_title('Fine-tuning with distillation('+str(len(fine_tuning_dist_distributions[0]))+')')
    params = {'legend.fontsize': 15,
            'axes.labelsize': 25,
            'axes.titlesize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 14,
            'axes.titlepad': 12,
            'axes.axisbelow': True}
    # plt.rc('axes', axisbelow=True)
    fig_acc.suptitle("SHAP. INT. GRAD. consistency", weight="bold")
    plt.yticks([.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    ax1.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    ax2.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    ax3.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    ax4.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    plt.rcParams.update(params)
    ax1.set_ylim(0.0, 1)
    ax2.set_ylim(0.0, 1)
    ax3.set_ylim(0.0, 1)
    ax4.set_ylim(0.0, 1)
    # plt.savefig('analysis/CompLing_results/int_grad.eps', bbox_inches = 'tight')
    fig_acc.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.8, hspace=None)
    plt.show()