import os
import shutil

CONNECTOME_FILES_PATH = '/run/media/i/ADATA HV620S/connectomes'
COMPLETED_JOBS_FOLDER = 'completed_jobs'
CONNECTOME_CSVS_FOLDER = 'connectomes'


def main():
    # Create connectomes folder, if it does not exist yet
    if not os.path.isdir(CONNECTOME_CSVS_FOLDER):
        os.mkdir(CONNECTOME_CSVS_FOLDER)
    for job_filename in os.listdir(COMPLETED_JOBS_FOLDER):
        ws = os.path.join(CONNECTOME_FILES_PATH, job_filename, 'weight_sum_connectome.csv')
        nws = os.path.join(CONNECTOME_FILES_PATH, job_filename, 'volume_normalized_weight_sum_connectome.csv')
        fa = os.path.join(CONNECTOME_FILES_PATH, job_filename, 'mean_FA_connectome.csv')
        ln = os.path.join(CONNECTOME_FILES_PATH, job_filename, 'mean_length_connectome.csv')
        ws_dst = os.path.join(CONNECTOME_CSVS_FOLDER, f'{job_filename}-ws.csv')
        nws_dst = os.path.join(CONNECTOME_CSVS_FOLDER, f'{job_filename}-nws.csv')
        fa_dst = os.path.join(CONNECTOME_CSVS_FOLDER, f'{job_filename}-fa.csv')
        ln_dst = os.path.join(CONNECTOME_CSVS_FOLDER, f'{job_filename}-ln.csv')
        for src, dst in [(ws, ws_dst), (nws, nws_dst), (fa, fa_dst), (ln, ln_dst)]:
            if not os.path.isfile(dst):
                shutil.copy(src, dst)


if __name__ == '__main__':
    main()
