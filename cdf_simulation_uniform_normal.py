import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats 
from scipy.stats import norm
from scipy.stats import uniform

for n in [10, 100]:
    # Iterations and bins of histograms:
    iterations = 10000
    bins       = 50

    # Initialization of maximum absolute deviation values D:
    D_uniform_val = []
    D_normal_val  = []

    for _ in range(iterations):
        # Uniform Samples in [0,1]:
        x = np.sort(np.random.uniform(0, 1, n))

        # Normal Distribution Samples N(0,1):
        y = np.sort(np.random.normal(0, 1, n))

        # Normalized Normal Distribution Samples to [0,1]:
        y_normalized = np.sort((y-np.min(y))/(np.max(y)-np.min(y)))

        #Empirical CDF values:
        cdf_values = np.linspace(0, 1, n)

        # Maximum absolute deviation D for each distribution:
        D_uniform = np.max(np.abs(cdf_values - uniform.cdf(x)))
        D_normal  = np.max(np.abs(cdf_values - norm.cdf(y)))
        D_uniform_val.append(np.sqrt(n) * D_uniform)
        D_normal_val.append( np.sqrt(n) * D_normal )

        if iterations == 1:
            #Plot the CDF of uniform and normal samples:
            plt.figure(figsize = (8, 6))
            plt.plot(x, cdf_values,     color="red", label = f"Uniform sample CDF", 
                     marker='o', linestyle = "", markersize = 3.5)
            plt.plot(x, uniform.cdf(x), color="black",  label = "Uniform Theoretical CDF")
            plt.legend()
            plt.title(f"Uniform CDF for n = {n} random values")
            plt.xlabel('Value', fontsize=14)
            plt.ylabel('Cumulative Probability', fontsize=14)
            plt.xlim(-0.05, 1.05)
            plt.ylim(-0.05, 1.05)
            plt.grid()
            plt.tight_layout()
            plt.show()

            plt.figure(figsize = (8, 6))
            plt.plot(y_normalized, cdf_values, color="red", label = "Normal sample N(0,1) CDF", 
                     marker='o', linestyle = "", markersize = 3.5)
            plt.plot(y_normalized, norm.cdf(y), color="black", label = "Normal Theoretical CDF")
            plt.legend()
            plt.title(f"Normal CDF for n = {n} random values")
            plt.xlabel('Value', fontsize=14)
            plt.ylabel('Cumulative Probability', fontsize=14)
            plt.xlim(-0.05, 1.05)
            plt.ylim(-0.05, 1.05)
            plt.grid()
            plt.tight_layout()
            plt.show()

            #Results for D value:
            print(f"The maximum absolute deviation for the uniform CDF with n = {n}: D = {D_uniform:.3f}")
            print(f"The maximum absolute deviation for the normal CDF with n = {n}:  D = {D_normal:.3f}")
    
    if iterations != 1:
        plt.figure(figsize = (8, 6))
        D_uni = plt.hist(D_uniform_val, bins, color="fuchsia", 
             density=True, alpha=1.0, edgecolor="indigo")
        plt.title(r"$\sqrt{{n}} \cdot D$ uniform values for {} iterations and n = {}".format(iterations, n))
        plt.xlabel(r'$\sqrt{{n}} \cdot D$ values', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.grid()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize = (8, 6))
        D_norm = plt.hist(D_normal_val, bins, color="fuchsia", 
             density=True, alpha=1.0, edgecolor="indigo")
        plt.title(r"$\sqrt{{n}} \cdot D$ uniform values for {} iterations and n = {}".format(iterations, n))
        plt.xlabel(r'$\sqrt{{n}} \cdot D$ values', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.grid()
        plt.tight_layout()
        plt.show()

        # Define the expected frequencies for a chi-square test:
        expected_uni  = np.ones(bins) * np.sum(D_uni[0])  / bins
        expected_norm = np.ones(bins) * np.sum(D_norm[0]) / bins

        # Perform the chi-square test:
        chi2_uni,   p_uni    = stats.chisquare( np.array(D_uni[0]),  expected_uni)
        chi2_norm,  p_norm   = stats.chisquare(np.array(D_norm[0]), expected_norm)

        #Results for chi2 test:
        print(f"For uniform with n = {n}, the chi2 test gives chi2 = {chi2_uni:.3f}  with p-value = {p_uni:.3f}")
        print(f"For normal with n = {n},  the chi2 test gives chi2 = {chi2_norm:.3f} with p-value = {p_norm:.3f}")
