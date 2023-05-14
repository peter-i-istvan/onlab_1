# Run this script:
# - from a conda environment having mrtrix3
# - having FSL installed on your system
# - having docker and the freesurfer/freesurfer:7.3.1 (or any specific version) image on your system:
#   docker pull freesurfer/freesurfer:7.3.1
# - Python >= 3.9
import os
import subprocess
import matplotlib.pyplot as plt
import pandas as pd

ANAT_ROOT_PATH = '/run/media/i/ADATA HV620S/rel3_dhcp_anat_pipeline'
DMRI_ROOT_PATH = '/run/media/i/ADATA HV620S/rel3_dhcp_dmri_eddy_pipeline'
TARGET_ROOT_PATH = '/run/media/i/ADATA HV620S/connectomes'
COMPLETED_JOBS_FOLDER = 'completed_jobs'


class DWISession:
    def __init__(self, session_source_root: str, session_target_root: str) -> None:
        '''Arguments:
        - session_source_root:
            The path to the dMRI measurement session.
            The path ends with sub-SUBID/ses-SESID/.
        - session_target_root:
            New files will be created here. We assume the root folder already exists.
        '''
        self.session_source_root = session_source_root
        self.session_target_root = session_target_root
        head, self.session_name = os.path.split(session_source_root)
        _, self.subject_name = os.path.split(head)

        self.source_dwi_folder = os.path.join(session_source_root, 'dwi')
        self.source_dwi_nii_gz = os.path.join(
            self.source_dwi_folder,
            f'{self.subject_name}_{self.session_name}_desc-preproc_dwi.nii.gz'
        )
        self.source_dwi_bvec = os.path.join(
            self.source_dwi_folder,
            f'{self.subject_name}_{self.session_name}_desc-preproc_dwi.bvec'
        )
        self.source_dwi_bval = os.path.join(
            self.source_dwi_folder,
            f'{self.subject_name}_{self.session_name}_desc-preproc_dwi.bval'
        )
        self.source_mask_nii_gz = os.path.join(
            self.source_dwi_folder,
            f'{self.subject_name}_{self.session_name}_desc-brain_mask.nii.gz'
        )
        self.source_FA_nii_gz = os.path.join(
            self.source_dwi_folder,
            f'{self.subject_name}_{self.session_name}_model-DTI_FA.nii.gz'
        )

        self.target_dwi_mif = os.path.join(self.session_target_root, 'dwi.mif')
        self.target_mask_mif = os.path.join(self.session_target_root, 'mask.mif')
        self.target_wm_response = os.path.join(self.session_target_root, 'wm.txt')
        self.target_gm_response = os.path.join(self.session_target_root, 'gm.txt')
        self.target_csf_response = os.path.join(self.session_target_root, 'csf.txt')
        self.target_csd_voxels = os.path.join(self.session_target_root, 'voxels.mif')
        self.target_wm_fod_mif = os.path.join(self.session_target_root, 'wmfod.mif')
        self.target_gm_fod_mif = os.path.join(self.session_target_root, 'gmfod.mif')
        self.target_csf_fod_mif = os.path.join(self.session_target_root, 'csffod.mif')
        self.target_concat_fod_mif = os.path.join(self.session_target_root, 'vf.mif')
        self.target_normalized_wm_fod_mif = os.path.join(self.session_target_root, 'wmfod_norm.mif')
        self.target_normalized_gm_fod_mif = os.path.join(self.session_target_root, 'gmfod_norm.mif')
        self.target_normalized_csf_fod_mif = os.path.join(self.session_target_root, 'csffod_norm.mif')

    def convert_dwi_nii_to_mif(self):
        command = [
            'mrconvert',
            f'{self.source_dwi_nii_gz}',
            f'{self.target_dwi_mif}',
            '-fslgrad',
            f'{self.source_dwi_bvec}',
            f'{self.source_dwi_bval}'
        ]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def convert_mask_to_mif(self):
        command = [
            'mrconvert',
            f'{self.source_mask_nii_gz}',
            f'{self.target_mask_mif}'
        ]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def dwi_to_response(self, view=False):
        command = [
            'dwi2response',
            'dhollander',
            f'{self.target_dwi_mif}',
            f'{self.target_wm_response}',
            f'{self.target_gm_response}',
            f'{self.target_csf_response}',
            '-voxels',
            f'{self.target_csd_voxels}'
        ]
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if process.returncode != 0:
            print('Something went wrong at calculating response function. Exiting...')
            return
        if view:
            command = ['mrview', f'{self.target_dwi_mif}', '-overlay.load', f'{self.target_csd_voxels}']
            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            assert process.returncode == 0

    def calculate_fod(self):
        command = [
            'dwi2fod',
            'msmt_csd',
            f'{self.target_dwi_mif}',
            '-mask',
            f'{self.target_mask_mif}',
            f'{self.target_wm_response}',
            f'{self.target_wm_fod_mif}',
            f'{self.target_gm_response}',
            f'{self.target_gm_fod_mif}',
            f'{self.target_csf_response}',
            f'{self.target_csf_fod_mif}',
        ]
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert process.returncode == 0, 'calculate_fod exited with errors'

    def combine_fods(self, view=False):
        command = [
            'mrconvert',
            '-coord',
            '3',
            '0',
            f'"{self.target_wm_fod_mif}"',  # duble quotes for safety
            '-',
            '|',
            'mrcat',
            f'"{self.target_csf_fod_mif}"',
            f'"{self.target_gm_fod_mif}"',
            '-',
            f'"{self.target_concat_fod_mif}"'
        ]
        # shell=True is not advised, but in this case, I could not find a clean solution to using pipes and - s
        # Ran into similar probl.: https://stackoverflow.com/questions/13332268/how-to-use-subprocess-command-with-pipes
        process = subprocess.run(
            ' '.join(command), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        assert process.returncode == 0, f'combine_fods ran into an error. STDERR: {process.stderr}'
        if view:
            command = ['mrview', f'{self.target_concat_fod_mif}', '-odf.load_sh', f'{self.target_wm_fod_mif}']
            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            assert process.returncode == 0

    def normalize_fods(self):
        '''Normalization is advised when doing group-level analisys.
        See:
        https://andysbrainbook.readthedocs.io/en/latest/MRtrix/MRtrix_Course/MRtrix_05_BasisFunctions.html#normalization
        '''
        command = [
            'mtnormalise',
            f'{self.target_wm_fod_mif}',
            f'{self.target_normalized_wm_fod_mif}',
            f'{self.target_gm_fod_mif}',
            f'{self.target_normalized_gm_fod_mif}',
            f'{self.target_csf_fod_mif}',
            f'{self.target_normalized_csf_fod_mif}',
            '-mask',
            f'{self.target_mask_mif}'
        ]
        subprocess.run(command, capture_output=True, check=True)


class AnatomicalSession:
    def __init__(self, session_source_root: str, session_target_root: str) -> None:
        self.session_source_root = session_source_root
        self.session_target_root = session_target_root
        head, self.session_name = os.path.split(session_source_root)
        _, self.subject_name = os.path.split(head)

        self.source_anat_folder = os.path.join(session_source_root, 'anat')
        self.source_t1w_nii_gz = os.path.join(
            self.source_anat_folder,
            f'{self.subject_name}_{self.session_name}_T1w.nii.gz'
        )
        self.source_regional_segmentation_nii_gz = os.path.join(
            self.source_anat_folder,
            f'{self.subject_name}_{self.session_name}_desc-drawem87_dseg.nii.gz'
        )

        self.target_t1w_mif = os.path.join(
            self.session_target_root,
            'T1w.mif'
        )
        self.target_5tt_nocoreg_mif = os.path.join(
            self.session_target_root,
            '5tt_nocoreg.mif'
        )

    def convert_t1w_nii_to_mif(self):
        command = [
            'mrconvert',
            f'{self.source_t1w_nii_gz}',
            f'{self.target_t1w_mif}'
        ]
        subprocess.run(command, capture_output=True, check=True)

    def five_tissue_segmentation(self, view=False):
        command = ['5ttgen', 'fsl', f'{self.target_t1w_mif}', f'{self.target_5tt_nocoreg_mif}']
        subprocess.run(command, capture_output=True, check=True)
        if view:
            subprocess.run(['mrview', f'{self.target_5tt_nocoreg_mif}'], capture_output=True, check=True)


class Session:
    '''It comes in handy starting from the coregistration step.'''
    def __init__(self, dwi_session: DWISession, anat_session: AnatomicalSession):
        self.dwi_session = dwi_session
        self.anat_session = anat_session
        assert self.dwi_session.session_target_root == self.anat_session.session_target_root
        self.session_target_root = self.dwi_session.session_target_root
        # Mean B0 image used for coregistration:
        self.target_mean_b0_mif = os.path.join(self.session_target_root, 'mean_b0.mif')
        self.target_mean_b0_nii = os.path.join(self.session_target_root, 'mean_b0.nii.gz')
        self.target_5tt_nocoreg_nii = os.path.join(self.session_target_root, '5tt_nocoreg.nii.gz')
        self.target_5tt_grey_nii = os.path.join(self.session_target_root, '5tt_vol0.nii.gz')
        # Transformation matrix was used to overlay the diffusion image on top of the grey matter segmentation
        # during coregistration (in MAT and TXT format):
        self.target_diff2struct_coregistration_mat = os.path.join(self.session_target_root, 'diff2struct_fsl.mat')
        self.target_diff2struct_coregistration_txt = os.path.join(self.session_target_root, 'diff2struct_mrtrix.txt')
        # Coregistered (=in DWI space) 5tt tissue type segmentation:
        self.target_5tt_coreg_mif = os.path.join(self.session_target_root, '5tt_coreg.mif')
        # Tissue boundary seed
        self.target_boundary_seed_mif = os.path.join(self.session_target_root, 'gmwmSeed_coreg.mif')
        # Streamlines: 1M (the tutorial said 10M, but that is an adult brain)
        self.target_streamlines_1M = os.path.join(self.session_target_root, 'tracks_1M.tck')
        # Subsample of streamlines for visualization purposes - can be skipped but good for diagnostic:
        self.target_streamlines_subsample = os.path.join(self.session_target_root, 'smallerTracks_500k.tck')
        # SIFT proportionality coefficient (debug purposes):
        self.target_sift_proportionality_coefficient = os.path.join(self.session_target_root, 'sift_mu.txt')
        # SIFT output coefficients - I did not find what is this supposed to be in the tcksift2 command help page.
        # Probably just for debug purposes:
        self.target_sift_output_coefficients = os.path.join(self.session_target_root, 'sift_coeffs.txt')
        # SIFT weights for each voxes - used at connectome creation
        self.target_sift_track_weights = os.path.join(self.session_target_root, 'sift_track_weights.txt')
        # The 4 types of connectomes generated:
        self.target_weight_sum_connectome = os.path.join(self.session_target_root, 'weight_sum_connectome.csv')
        self.target_volume_normalized_weight_sum_connectome = os.path.join(
            self.session_target_root, 'volume_normalized_weight_sum_connectome.csv'
        )
        self.target_mean_FA_per_streamline_sample = os.path.join(
            self.session_target_root, 'mean_FA_per_streamline.csv'
        )
        self.target_mean_FA_connectome = os.path.join(
            self.session_target_root, 'mean_FA_connectome.csv'
        )
        self.target_mean_length_connectome = os.path.join(
            self.session_target_root, 'mean_length_connectome.csv'
        )

    def __get_mean_b0(self):
        '''Calculates mean signal values for bval=0 slices along the time axis.
        Part of the coregistration step. Called by Session.coregister_dwi_anat()'''
        command = [
            'dwiextract',
            f'"{self.dwi_session.target_dwi_mif}"',
            '-',
            '-bzero',
            '|',
            'mrmath',
            '-',
            'mean',
            f'"{self.target_mean_b0_mif}"',
            '-axis',
            '3'
        ]
        # shell=True is not advised, but in this case, I could not find a clean solution to using pipes and - s
        # Ran into similar probl.: https://stackoverflow.com/questions/13332268/how-to-use-subprocess-command-with-pipes
        process = subprocess.run(
            ' '.join(command), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        assert process.returncode == 0, f'get_mean_b0 ran into an error. STDERR: {process.stderr}'

    def __mean_b0_and_5tt_to_nii(self):
        convert_mean_command = ['mrconvert', f'{self.target_mean_b0_mif}', f'{self.target_mean_b0_nii}']
        convert_5tt_command = [
            'mrconvert', f'{self.anat_session.target_5tt_nocoreg_mif}', f'{self.target_5tt_nocoreg_nii}'
        ]
        subprocess.run(convert_mean_command, capture_output=True, check=True)
        subprocess.run(convert_5tt_command, capture_output=True, check=True)

    def __extract_grey_matter(self):
        command = ['fslroi', f'{self.target_5tt_nocoreg_nii}', f'{self.target_5tt_grey_nii}', '0', '1']
        subprocess.run(command, capture_output=True, check=True)

    def __coregister(self):
        generate_trf_command = [
            'flirt',
            '-in',
            f'{self.target_mean_b0_nii}',
            '-ref',
            f'{self.target_5tt_grey_nii}',
            '-interp',
            'nearestneighbour',
            '-dof',
            '6',
            '-omat',
            f'{self.target_diff2struct_coregistration_mat}'
        ]
        convert_trf_command = [
            'transformconvert',
            f'{self.target_diff2struct_coregistration_mat}',
            f'{self.target_mean_b0_nii}',
            f'{self.target_5tt_nocoreg_nii}',
            'flirt_import',
            f'{self.target_diff2struct_coregistration_txt}'
        ]
        # Generate coregistration of 5tt via the inverse struct -> diff transformation
        # We used the anatomical image as reference only because the higher spatial resolution
        # and sharper distinction btw. tissue types. The transformation, however, is needed in the
        # reverse direction.
        inverse_trf_command = [
            'mrtransform',
            f'{self.anat_session.target_5tt_nocoreg_mif}',
            '-linear',
            f'{self.target_diff2struct_coregistration_txt}',
            '-inverse',
            f'{self.target_5tt_coreg_mif}'
        ]
        subprocess.run(generate_trf_command, capture_output=True, check=True)
        subprocess.run(convert_trf_command, capture_output=True, check=True)
        subprocess.run(inverse_trf_command, capture_output=True, check=True)

    def coregister_dwi_anat(self, view=False):
        self.__get_mean_b0()
        self.__mean_b0_and_5tt_to_nii()
        self.__extract_grey_matter()
        self.__coregister()
        if view:
            command = [
                'mrview',
                f'{self.dwi_session.target_dwi_mif}',
                '-overlay.load',
                f'{self.anat_session.target_5tt_nocoreg_mif}',
                '-overlay.colourmap',
                '2',
                '-overlay.load',
                f'{self.target_5tt_coreg_mif}',
                '-overlay.colourmap',
                '1'
            ]
            subprocess.run(command, capture_output=True, check=True)

    def create_gmwm_seed_boundary(self, view=False):
        command = ['5tt2gmwmi', f'{self.target_5tt_coreg_mif}', f'{self.target_boundary_seed_mif}']
        subprocess.run(command, capture_output=True, check=True)
        if view:
            command = [
                'mrview', f'{self.dwi_session.target_dwi_mif}', '-overlay.load', f'{self.target_boundary_seed_mif}'
            ]
            subprocess.run(command, capture_output=True, check=True)

    def generate_streamlines(self, number: int = 1_000_000, nthreads: int = 6, view=False):
        command = [
            'tckgen',
            '-act',
            f'{self.target_5tt_coreg_mif}',
            '-backtrack',
            '-seed_gmwmi',
            f'{self.target_boundary_seed_mif}',
            '-nthreads',
            str(nthreads),
            '-maxlength',
            '250',
            '-cutoff',
            '0.06',
            '-select',
            str(number),
            f'{self.dwi_session.target_normalized_wm_fod_mif}',
            f'{self.target_streamlines_1M}'
        ]
        subprocess.run(command, capture_output=True, check=True)
        if view:
            sample_command = [
                'tckedit', f'{self.target_streamlines_1M}', '-number', '500k', f'{self.target_streamlines_subsample}'
            ]
            view_command = [
                'mrview',
                f'{self.dwi_session.target_dwi_mif}',
                '-tractography.load',
                f'{self.target_streamlines_subsample}'
            ]
            subprocess.run(sample_command, capture_output=True, check=True)
            subprocess.run(view_command, capture_output=True, check=True)

    def sift_filltering(self, nthreads: int = 6):
        '''From Andy's Brain Book:
        Certain tracts can be over-represented by the amount of streamlines that pass through them
        not necessarily because they contain more fibers, but because the fibers tend to all be orientated
        in the same direction.
        To counter-balance this overfitting, the command tcksift2 will create a text file
        containing weights for each voxel in the brain. The output from the command, can be used [...]
        to create a matrix [...] known as a connectome - which will weight each ROI.

        From the command's help page:
        Filter a whole-brain fibre-tracking data set such that the streamline densities
        match the FOD lobe integrals'''
        command = [
            'tcksift2',
            '-act',
            f'{self.target_5tt_coreg_mif}',
            '-out_mu',
            f'{self.target_sift_proportionality_coefficient}',
            '-out_coeffs',
            f'{self.target_sift_output_coefficients}',
            '-nthreads',
            str(nthreads),
            f'{self.target_streamlines_1M}',
            f'{self.dwi_session.target_normalized_wm_fod_mif}',
            f'{self.target_sift_track_weights}'
        ]
        subprocess.run(command, capture_output=True, check=True)

    def __view_connectomes(self):
        ws = pd.read_csv(self.target_weight_sum_connectome, header=None)
        nws = pd.read_csv(self.target_volume_normalized_weight_sum_connectome, header=None)
        fa = pd.read_csv(self.target_mean_FA_connectome, header=None)
        ln = pd.read_csv(self.target_mean_length_connectome, header=None)
        for i, df in enumerate([ws, nws, fa, ln]):
            plt.matshow(df.to_numpy(), i)
            plt.show()

    def generate_connectome(self, view=False):
        generate_weight_sum_command = [
            'tck2connectome',
            '-symmetric',
            '-zero_diagonal',
            '-tck_weights_in',
            f'{self.target_sift_track_weights}',
            f'{self.target_streamlines_1M}',
            f'{self.anat_session.source_regional_segmentation_nii_gz}',
            f'{self.target_weight_sum_connectome}'
        ]
        generate_normalized_weight_sum_command = [
            'tck2connectome',
            '-symmetric',
            '-zero_diagonal',
            '-scale_invnodevol',
            '-tck_weights_in',
            f'{self.target_sift_track_weights}',
            f'{self.target_streamlines_1M}',
            f'{self.anat_session.source_regional_segmentation_nii_gz}',
            f'{self.target_volume_normalized_weight_sum_connectome}'
        ]
        # Mean FA connectome: two-step process
        sample_mean_FA_command = [
            'tcksample',
            f'{self.target_streamlines_1M}',
            f'{self.dwi_session.source_FA_nii_gz}',
            f'{self.target_mean_FA_per_streamline_sample}',
            '-stat_tck',
            'mean'
        ]
        generate_mean_FA_connectome_command = [
            'tck2connectome',
            '-symmetric',
            '-zero_diagonal',
            f'{self.target_streamlines_1M}',
            f'{self.anat_session.source_regional_segmentation_nii_gz}',
            f'{self.target_mean_FA_connectome}',
            '-scale_file',
            f'{self.target_mean_FA_per_streamline_sample}',
            '-stat_edge',
            'mean'
        ]
        generate_mean_length = [
            'tck2connectome',
            '-symmetric',
            '-zero_diagonal',
            f'{self.target_streamlines_1M}',
            f'{self.anat_session.source_regional_segmentation_nii_gz}',
            f'{self.target_mean_length_connectome}',
            '-scale_length',
            '-stat_edge',
            'mean'
        ]
        commands_list = [
            generate_weight_sum_command,
            generate_normalized_weight_sum_command,
            sample_mean_FA_command,
            generate_mean_FA_connectome_command,
            generate_mean_length,
        ]
        for command in commands_list:
            subprocess.run(command, capture_output=True, check=True, text=True)
        if view:
            self.__view_connectomes()


class Job:
    '''Encompasses the generation of the connectomes for a single subject-session pair.'''
    def __init__(self, subject_name: str, session_name: str) -> None:
        self.completed_jobs_file = os.path.join(COMPLETED_JOBS_FOLDER, f'{subject_name}-{session_name}')
        # It is important to omit the last '/' of a folder path
        # Otherwise os.ptah.split will generate a '' at the tail
        self.dwi_session_source_root = os.path.join(DMRI_ROOT_PATH, subject_name, session_name)
        self.anat_session_source_root = os.path.join(ANAT_ROOT_PATH, subject_name, session_name)
        self.session_target_root = os.path.join(TARGET_ROOT_PATH, f'{subject_name}-{session_name}')

    def __is_already_done(self):
        return os.path.isfile(self.completed_jobs_file)

    def __mark_complete(self):
        subprocess.run(['touch', self.completed_jobs_file])

    def run(self):
        # I. Convert dwi NIFTI to .mif file
        # II. Convert mask NIFTI to .mif file
        # III. Run spherical deconv. to compute signal response in each tissue type, based on diffusion signal:
        # - dhollander algorithm for multi-shell (bval) multi-tissue (gm, wm, csf)
        # - additional output voxels.mif highlights which voxels were used to construct the signal response
        #   - R=CSF, G=GM, B=WM
        # IV. Calculate FOD based on the signal responses (basis functions) above
        # V. (Optional) Concatenate and visualize FODs
        # VI. Normalize FODs for proper group level usage (Do I need it in this use case?)
        # VII. Convert anatomical T1w NIFTI to .mif file
        # VIII. Tissue segmentation in order to determine wm/gm boundary (interface)
        # - The MRTrix guide provides a 5 tissue segmentation example: GM, Subcortical GM, WM, CSF, Pathological
        # - dHCP provides a 9 tissue segmentation in a similar manner,
        #   but mapping some tissues to the ones above is not trivial for me, so I will ditch that for the moment.
        #   The official MRTrix docs encourage using 5TT.
        # - I will use 5ttgen which uses FSL for this goal
        # IX. Coregistration (divided into substeps)
        # X. Apply tissue bundary sgementation
        # XI. Generate streamlines (ACT)
        # XII. Filter (SIFT) to combat overfitting
        # XIII. Generate connectome
        # Mark job completed
        if self.__is_already_done():
            print('Process already completed. Skipping...')
            return

        dwi = DWISession(
            session_source_root=self.dwi_session_source_root,
            session_target_root=self.session_target_root
        )
        anat = AnatomicalSession(
            session_source_root=self.anat_session_source_root,
            session_target_root=self.session_target_root
        )
        session = Session(dwi_session=dwi, anat_session=anat)
        dwi.convert_dwi_nii_to_mif()
        dwi.convert_mask_to_mif()
        dwi.dwi_to_response()
        dwi.calculate_fod()
        dwi.combine_fods(view=False)
        dwi.normalize_fods()
        anat.convert_t1w_nii_to_mif()
        anat.five_tissue_segmentation(view=False)
        session.coregister_dwi_anat(view=False)
        session.create_gmwm_seed_boundary(view=False)
        session.generate_streamlines(number=1_000_000, nthreads=6, view=False)
        session.sift_filltering(nthreads=6)
        session.generate_connectome(view=True)
        self.__mark_complete()


def main():
    if not os.path.isdir(COMPLETED_JOBS_FOLDER):
        os.mkdir(COMPLETED_JOBS_FOLDER)
    # Job completion will be marked with a single 0B file called {subject}_{session} via 'touch' command
    job = Job(subject_name='sub-CC00063AN06', session_name='ses-15102')
    job.run()


if __name__ == '__main__':
    main()
