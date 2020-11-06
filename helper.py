import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import scipy.stats as scs

def mde(p_A, p_B):
    return p_B-p_A

# Calculates the pooled probability for two samples
def pooled_prob(N_A, N_B, X_A, X_B):
    return (X_A + X_B) / (N_A + N_B)

# Calculates the standard deviation for the pooled probability of 2 samples
def std_pooled_prob(N_A, N_B, X_A, X_B):
    p_hat = pooled_prob(N_A, N_B, X_A, X_B)
    SE = np.sqrt(p_hat * (1 - p_hat) * (1 / N_A + 1 / N_B))
    return SE

# Calculates z for a given significance level, for one or two tailed distribution
def z_val(sig_level = 0.05, two_tailed = True):
    
    # Creates distribution for z
    z_dist = scs.norm()
    
    # Calculates the are for one or two-tailed
    if two_tailed:
        sig_level = sig_level/2
        area = 1 - sig_level
    else:
        area = 1 - sig_level

    # Calculates z
    z = z_dist.ppf(area)

    return z

# Calculates the confidence interval
def confidence_interval(sample_mean = 0, sample_std = 1, sample_size = 1, sig_level = 0.05):
    
    # Calculates z
    z = z_val(sig_level)

    # Calculates limits to the left and right
    left = sample_mean - z * sample_std / np.sqrt(sample_size)
    right = sample_mean + z * sample_std / np.sqrt(sample_size)

    return (left, right)

# Plot the confidence interval for two-tailed
def plot_CI(ax, 
            mu, 
            s, 
            sig_level = 0.05, 
            color = 'grey'):

    # Confidence interval
    left, right = confidence_interval(sample_mean = mu, sample_std = s, sig_level = sig_level)
    
    # Add intervals in the graph
    ax.axvline(left, c = color, linestyle = '--', alpha = 0.5)
    ax.axvline(right, c = color, linestyle = '--', alpha = 0.5)

# Plot a normal distribution
def plot_norm_dist(ax, 
                   mu, 
                   std, 
                   with_CI = False, 
                   sig_level = 0.05, 
                   label = None):

    # Generate values for x
    x = np.linspace(mu - 12 * std, mu + 12 * std, 1000)
    
    # Creates a normal distribution
    y = scs.norm(mu, std).pdf(x)
    
    # Plot
    ax.plot(x, y, label = label)

    # Add confidence interval
    if with_CI:
        plot_CI(ax, mu, std, sig_level = sig_level)

# Plot H0 distribution
def plot_H0(ax, stderr):
    plot_norm_dist(ax, 0, stderr, label = "H0 - Null Hypothesis")
    plot_CI(ax, mu = 0, s = stderr, sig_level = 0.05)

# Plot H1 distribution
def plot_H1(ax, stderr, d_hat):
    plot_norm_dist(ax, d_hat, stderr, label = "H1 - Alternative Hypothesis")

#  Fill in the limit betwwen the upper ci and the alternative hypothesis distribution
def show_area(ax, d_hat, stderr, sig_level):

    # CI
    left, right = confidence_interval(sample_mean = 0, sample_std = stderr, sig_level = sig_level)
    
    # x
    x = np.linspace(-12 * stderr, 12 * stderr, 1000)
    
    # H0
    null = ab_dist(stderr, 'control')
    
    # H1
    alternative = ab_dist(stderr, d_hat, 'test')

    # If area = power
    # Fill in the limit betwwen the upper ci and the alternative hypothesis distribution
    ax.fill_between(x, 0, alternative.pdf(x), color = 'green', alpha = 0.25, where = (x > right))
    ax.text(-3 * stderr, null.pdf(0), 'power = {0:.3f}'.format(1 - alternative.cdf(right)), 
                fontsize = 12, ha = 'right', color = 'k')

# Função que retorna um objeto de distribuição dependendo do tipo de grupo
def ab_dist(stderr, d_hat = 0, group_type = 'control'):

    # Verify group type
    if group_type == 'control':
        sample_mean = 0
    elif group_type == 'test':
        sample_mean = d_hat

    # Create a normal distribution
    dist = scs.norm(sample_mean, stderr)
    return dist

# Calculates p-value
def p_val(N_A, N_B, p_A, p_B):
    return scs.binom(N_A, p_A).pmf(p_B * N_B)

# Plot AB distributions
def abplot_func(N_A, 
                N_B, 
                bcr, 
                d_hat, 
                sig_level = 0.05, 
                show_p_value = False,
                show_legend = True):
   
    # Define plot area
    fig, ax = plt.subplots(figsize = (14, 8))

    # Define parameters to calculated the standard pooled error

    X_A = bcr * N_A
    X_B = (bcr + d_hat) * N_B
    stderr = std_pooled_prob(N_A, N_B, X_A, X_B)

    # Plot null and alternative distributions
    plot_H0(ax, stderr)
    plot_H1(ax, stderr, d_hat)

    # Set plot limits
    ax.set_xlim(-8 * stderr, 8 * stderr)

    # Adjust graph and fill areas
    show_area(ax, d_hat, stderr, sig_level)

    # Show p-value
    if show_p_value:
        null = ab_dist(stderr, 'control')
        p_value = p_val(N_A, N_B, bcr, bcr + d_hat)
        ax.text(3 * stderr, null.pdf(0), 'p-value = {0:.4f}'.format(p_value), fontsize = 14, ha = 'left')

    # Show legend
    if show_legend:
        plt.legend()

    plt.xlabel('d')
    plt.ylabel('PDF')
    plt.show()

# Includes z value in the plot
def zplot(area = 0.95, two_tailed = True, align_right = False):

    # Define plot ares
    fig = plt.figure(figsize = (12, 6))
    ax = fig.subplots()
    
    # Created normal distribution
    norm = scs.norm()
    
    # Creates x
    x = np.linspace(-5, 5, 1000)
    y = norm.pdf(x)

    ax.plot(x, y)

    # Fill in area if two-tailed
    if two_tailed:
        left = norm.ppf(0.5 - area / 2)
        right = norm.ppf(0.5 + area / 2)
        ax.vlines(right, 0, norm.pdf(right), color = 'grey', linestyle = '--')
        ax.vlines(left, 0, norm.pdf(left), color = 'grey', linestyle = '--')

        ax.fill_between(x, 0, y, color = 'grey', alpha = 0.25, where = (x > left) & (x < right))
        
        plt.xlabel('z')
        plt.ylabel('PDF')
        plt.text(left, norm.pdf(left), "z = {0:.3f}".format(left), 
                 fontsize = 12, 
                 rotation = 90, 
                 va = "bottom", 
                 ha = "right")
        plt.text(right, norm.pdf(right), "z = {0:.3f}".format(right), 
                 fontsize = 12, 
                 rotation = 90, 
                 va = "bottom", 
                 ha = "left")
    
    # Fill in area if one-tailed
    else:
        # Align to right
        if align_right:
            left = norm.ppf(1-area)
            ax.vlines(left, 0, norm.pdf(left), color = 'grey', linestyle = '--')
            ax.fill_between(x, 0, y, color = 'grey', alpha = 0.25, where = x > left)
            plt.text(left, norm.pdf(left), "z = {0:.3f}".format(left), 
                     fontsize = 12, 
                     rotation = 90, 
                     va = "bottom", 
                     ha = "right")
        
        # Align to left
        else:
            right = norm.ppf(area)
            ax.vlines(right, 0, norm.pdf(right), color = 'grey', linestyle = '--')
            ax.fill_between(x, 0, y, color = 'grey', alpha = 0.25, where = x < right)
            plt.text(right, norm.pdf(right), "z = {0:.3f}".format(right), 
                     fontsize = 12, 
                     rotation = 90, 
                     va = "bottom", 
                     ha = "left")

    # Add text
    plt.text(0, 0.1, "Shaded area = {0:.3f}".format(area), fontsize = 12, ha = 'center')
    
    # Labels
    plt.xlabel('z')
    plt.ylabel('PDF')

    plt.show()

# Calculates the minimum sample size
def minimum_sample_size(N_A, N_B, p_A, p_B, power = 0.8, sig_level = 0.05, two_sided = False):
   
    k = N_A/N_B
    
    # Normal distribution
    standard_norm = scs.norm(0, 1)

    # Calculates z for power
    Z_beta = standard_norm.ppf(power)

    # Calculates z for alpha
    if two_sided == True:
        Z_alpha = standard_norm.ppf(1-sig_level/2)
    else:
        Z_alpha = standard_norm.ppf(1-sig_level/2)

    # Pooled probability
    pooled_prob = (p_A + p_B) / 2

    # Calculates the minimum sample size
    min_N = (2 * pooled_prob * (1 - pooled_prob) * (Z_beta + Z_alpha)**2 / mde(p_A,p_B)**2)    

    return min_N