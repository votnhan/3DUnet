import pandas as pd
import matplotlib.pyplot as plt
import os
from pandas import DataFrame

log_file = 'training.log'
log_file_mdf = 'training_mdf.log'

def edit_epoch_collumn_in_logfile(inp_file, outp_file):
    csv_inp = pd.read_csv(log_file)
    n_rows = len(csv_inp['epoch'])
    ep = [x for x in range(n_rows)]
    csv_outp = DataFrame({'epoch':ep, 'loss': csv_inp['loss'], 'val_loss': csv_inp['val_loss']}, 
                    columns=['epoch', 'loss', 'val_loss'])
    export_csv = csv_outp.to_csv(outp_file, header=True, index=None)

def visualize_training_process(logfile):
    if os.path.exists(logfile):
        training_df = pd.read_csv(logfile).set_index('epoch')
        plt.plot(training_df['loss'].values, label='Train loss')
        plt.plot(training_df['val_loss'].values, label='Validation loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.xlim((0, len(training_df.index)))
        plt.legend(loc='upper right')
        plt.savefig('loss_graph_main.png')
        return 'Done'

    return 'Log File Not Found'


edit_epoch_collumn_in_logfile(log_file, log_file_mdf)
visualize_training_process(log_file_mdf)