import sys
import cma

# python -m optimization.plot_cma './data/cma/trial_180625_1_'
if __name__ == '__main__':
    cma.plot(sys.argv[1])
    input()