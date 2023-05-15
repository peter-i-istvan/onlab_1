import os
import shutil

TARGET_ROOT_PATH = '/run/media/i/ADATA HV620S/connectomes'
COMPLETED_JOBS_FOLDER = 'completed_jobs'
CONNECTOME_CSVS_FOLDER = 'connectomes'


def main():
    # Create connectomes folder, if it does not exist yet
    if not os.path.isdir(CONNECTOME_CSVS_FOLDER):
        os.mkdir(CONNECTOME_CSVS_FOLDER)
    for job_filename in os.listdir(COMPLETED_JOBS_FOLDER):
        ws = os.path.join(TARGET_ROOT_PATH, job_filename, 'weight_sum_connectome.csv')
        nws = os.path.join(TARGET_ROOT_PATH, job_filename, 'volume_normalized_weight_sum_connectome.csv')
        fa = os.path.join(TARGET_ROOT_PATH, job_filename, 'mean_FA_connectome.csv')
        ln = os.path.join(TARGET_ROOT_PATH, job_filename, 'mean_length_connectome.csv')
        shutil.copy(ws, os.path.join(CONNECTOME_CSVS_FOLDER, f'{job_filename}-ws.csv'))
        shutil.copy(nws, os.path.join(CONNECTOME_CSVS_FOLDER, f'{job_filename}-nws.csv'))
        shutil.copy(fa, os.path.join(CONNECTOME_CSVS_FOLDER, f'{job_filename}-fa.csv'))
        shutil.copy(ln, os.path.join(CONNECTOME_CSVS_FOLDER, f'{job_filename}-ln.csv'))


if __name__ == '__main__':
    main()
