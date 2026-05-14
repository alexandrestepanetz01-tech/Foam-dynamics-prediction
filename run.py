import main, Split_Configurations, Data_Generator, plot_conf,  AverageD2min
import multiprocessing as mp


def run(d, cnfs):
    mp.freeze_support()
    main.mlp_main(d, cnfs)
    Split_Configurations.main(d, cnfs)
    Data_Generator.main(d, cnfs)
    plot_conf.main(d, cnfs)
    AverageD2min.main(d, cnfs)

if __name__ == "__main__":
    d = 0.9
    cnfs = [856]
    mp.freeze_support()
    run(d, cnfs)
